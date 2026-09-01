#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESSD teacher + prototype-library construction.

Logic implemented here:
  HQ image patch --E_img^T--> v_i^T
  component mask --E_mask^T--> g_i^T
  teacher training: L_align + eta_con * L_contrast
  frozen mask embeddings --per-entity clustering--> prototype bank M_e

The saved prototype library is shared by the HQ teacher and LQ student when
computing prototype-association distributions during ESSD training.
"""

import argparse
import os
import pickle
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


DEFAULT_IMG_SIZE = 128
DEFAULT_PATCH = 16
DEFAULT_BATCH = 32
DEFAULT_EMBED = 256
DEFAULT_K = 8


class TeacherModel(nn.Module):
    """HQ teacher defining the shared structural embedding space."""

    def __init__(self, img_dim=3, mask_dim=1, embed_dim=DEFAULT_EMBED):
        super().__init__()
        self.output_dim = embed_dim

        self.img_encoder = nn.Sequential(
            nn.Conv2d(img_dim, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.img_proj = nn.Linear(128, embed_dim)

        self.mask_encoder = nn.Sequential(
            nn.Conv2d(mask_dim, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.mask_proj = nn.Linear(64, embed_dim)

    def encode_image(self, img_patch):
        feat = self.img_encoder(img_patch).flatten(1)
        return F.normalize(self.img_proj(feat), dim=-1)

    def encode_mask(self, mask_patch):
        feat = self.mask_encoder(mask_patch).flatten(1)
        return F.normalize(self.mask_proj(feat), dim=-1)

    def forward(self, img_patch, mask_patch):
        return self.encode_image(img_patch), self.encode_mask(mask_patch)


def parse_yolov8_seg_line(line):
    vals = line.strip().split()
    if len(vals) < 7:
        return None
    try:
        cls = int(vals[0])
        coords = [float(x) for x in vals[1:]]
        if len(coords) % 2:
            coords = coords[:-1]
        if len(coords) < 6:
            return None
        poly = np.asarray(coords, dtype=np.float32).reshape(-1, 2)
        return cls, poly
    except (TypeError, ValueError):
        return None


def polygon_to_mask(poly_norm, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    if poly_norm is None:
        return mask
    pts = np.asarray(poly_norm, dtype=np.float32).reshape(-1, 2).copy()
    if pts.shape[0] < 3:
        return mask
    pts[:, 0] *= w
    pts[:, 1] *= h
    pts = np.round(pts).astype(np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


class YOLOSegDataset(Dataset):
    """Loads HQ images and component-level YOLO-seg labels."""

    def __init__(self, img_dir, label_dir, img_size=DEFAULT_IMG_SIZE):
        self.img_paths = sorted(
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
        )
        self.label_dir = label_dir
        self.img_size = img_size

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        stem = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(self.label_dir, stem + ".txt")

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Cannot read {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC)

        instances = []
        if os.path.exists(label_path):
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    parsed = parse_yolov8_seg_line(line)
                    if parsed is not None:
                        instances.append(parsed)
        return image, instances, img_path


def identity_collate(batch):
    return batch[0]


def collect_paired_component_patches(
    image,
    instances,
    patch_size=DEFAULT_PATCH,
    stride=None,
    min_area_ratio=0.02,
    dilate=0,
):
    """Return aligned (entity_id, HQ image patch, mask patch, xy) tuples."""
    if stride is None:
        stride = patch_size

    h, w = image.shape[:2]
    out = []
    for entity_id, poly in instances:
        mask = polygon_to_mask(poly, h, w)
        if dilate > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1)
            )
            mask = cv2.dilate(mask, kernel)

        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                m = mask[y:y + patch_size, x:x + patch_size]
                if np.count_nonzero(m) / float(patch_size * patch_size) < min_area_ratio:
                    continue
                img_patch = image[y:y + patch_size, x:x + patch_size]
                out.append((entity_id, img_patch, m.copy(), (x, y)))
    return out


class PatchPairDataset(Dataset):
    def __init__(self, patch_list):
        self.patch_list = patch_list

    def __len__(self):
        return len(self.patch_list)

    def __getitem__(self, idx):
        _, img_np, mask_np, _ = self.patch_list[idx]
        img_t = torch.from_numpy(img_np.astype(np.float32) / 255.0).permute(2, 0, 1)
        mask_t = torch.from_numpy(mask_np.astype(np.float32) / 255.0).unsqueeze(0)
        return img_t, mask_t


def teacher_loss(v, g, temperature=0.07, contrast_weight=0.5):
    """L_T = L_align + eta_con * L_contrast."""
    l_align = F.mse_loss(v, g)
    logits = v @ g.t() / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    l_con = 0.5 * (
        F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)
    )
    return l_align + contrast_weight * l_con, l_align.detach(), l_con.detach()


def train_teacher(
    teacher,
    patch_list,
    device,
    epochs=20,
    batch_size=64,
    lr=1e-4,
    temperature=0.07,
    contrast_weight=0.5,
):
    if not patch_list:
        raise RuntimeError("No valid component patches were collected.")

    loader = DataLoader(
        PatchPairDataset(patch_list), batch_size=batch_size, shuffle=True, num_workers=4
    )
    optimizer = torch.optim.Adam(teacher.parameters(), lr=lr)
    teacher.to(device)

    for epoch in range(epochs):
        teacher.train()
        running = 0.0
        for img_patch, mask_patch in tqdm(loader, desc=f"Teacher {epoch + 1}/{epochs}"):
            img_patch = img_patch.to(device, non_blocking=True)
            mask_patch = mask_patch.to(device, non_blocking=True)
            v, g = teacher(img_patch, mask_patch)
            loss, _, _ = teacher_loss(v, g, temperature, contrast_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += loss.item()
        print(f"Epoch {epoch + 1}: loss={running / max(len(loader), 1):.6f}")
    return teacher


@torch.no_grad()
def extract_mask_embeddings(teacher, patch_list, device, batch_size=256):
    """Extract teacher mask-space embeddings grouped by aircraft entity."""
    teacher.eval().to(device)
    grouped = defaultdict(list)

    for start in tqdm(range(0, len(patch_list), batch_size), desc="Mask embeddings"):
        chunk = patch_list[start:start + batch_size]
        masks = np.stack([item[2] for item in chunk]).astype(np.float32) / 255.0
        mask_t = torch.from_numpy(masks).unsqueeze(1).to(device)
        emb = teacher.encode_mask(mask_t).cpu().numpy()

        for item, e in zip(chunk, emb):
            entity_id, _, mask_np, coord = item
            grouped[str(entity_id)].append(
                {
                    "embedding": e.astype(np.float32),
                    "mask": mask_np,
                    "coord": tuple(coord),
                }
            )
    return grouped


def _kmeans_np(x, k, seed=0, max_iter=100):
    """Small dependency-free k-means used only for prototype construction."""
    x = np.asarray(x, dtype=np.float32)
    if len(x) == 0:
        raise ValueError("Cannot cluster an empty embedding set.")
    k = min(int(k), len(x))
    rng = np.random.default_rng(seed)
    centers = x[rng.choice(len(x), size=k, replace=False)].copy()

    for _ in range(max_iter):
        dist = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
        labels = dist.argmin(axis=1)
        new_centers = centers.copy()
        for j in range(k):
            members = x[labels == j]
            if len(members):
                new_centers[j] = members.mean(axis=0)
        new_centers /= np.linalg.norm(new_centers, axis=1, keepdims=True) + 1e-8
        if np.allclose(new_centers, centers, atol=1e-5):
            centers = new_centers
            break
        centers = new_centers
    return centers, labels


def build_prototype_library(embeddings_per_entity, k_per_entity=DEFAULT_K, seed=0):
    """Construct M_e={mu_e,k}; prototypes are mask-derived teacher embeddings."""
    prototype_lib = {}
    for entity_id, items in embeddings_per_entity.items():
        if not items:
            continue
        x = np.stack([it["embedding"] for it in items])
        x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
        centers, labels = _kmeans_np(x, k_per_entity, seed=seed)

        prototypes = []
        for k, center in enumerate(centers):
            members = np.where(labels == k)[0]
            if len(members):
                sims = x[members] @ center
                medoid_idx = members[int(np.argmax(sims))]
                exemplar = items[medoid_idx]
                centroid = exemplar.get("coord", None)
                mask = exemplar.get("mask", None)
            else:
                centroid, mask = None, None
            prototypes.append(
                {
                    "embedding": center.astype(np.float32),
                    "centroid": centroid,
                    "mask": mask,
                    "count": int(len(members)),
                }
            )
        prototype_lib[str(entity_id)] = prototypes
    return prototype_lib


def save_prototype_library(prototype_lib, out_path):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(prototype_lib, f)
    print(f"Saved prototype library to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", required=True, help="HQ training images")
    parser.add_argument("--label_dir", required=True, help="component-level YOLO-seg labels")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--img_size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--patch_size", type=int, default=DEFAULT_PATCH)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--embed_dim", type=int, default=DEFAULT_EMBED)
    parser.add_argument("--num_prototypes", type=int, default=DEFAULT_K)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--contrast_weight", type=float, default=0.5)
    parser.add_argument("--min_area_ratio", type=float, default=0.02)
    parser.add_argument("--dilate", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = YOLOSegDataset(args.img_dir, args.label_dir, args.img_size)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4, collate_fn=identity_collate)

    patches = []
    for image, instances, _ in tqdm(loader, desc="Collect component patches"):
        patches.extend(
            collect_paired_component_patches(
                image,
                instances,
                patch_size=args.patch_size,
                stride=args.stride,
                min_area_ratio=args.min_area_ratio,
                dilate=args.dilate,
            )
        )
    print(f"Collected {len(patches)} component patches.")

    teacher = TeacherModel(embed_dim=args.embed_dim).to(device)
    teacher = train_teacher(
        teacher,
        patches,
        device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        temperature=args.temperature,
        contrast_weight=args.contrast_weight,
    )

    ckpt_path = os.path.join(args.out_dir, "teacher.pth")
    torch.save(teacher.state_dict(), ckpt_path)
    print(f"Saved teacher checkpoint to {ckpt_path}")

    embeddings = extract_mask_embeddings(teacher, patches, device)
    prototype_lib = build_prototype_library(
        embeddings, k_per_entity=args.num_prototypes
    )
    save_prototype_library(
        prototype_lib, os.path.join(args.out_dir, "prototype_lib.pkl")
    )


if __name__ == "__main__":
    main()

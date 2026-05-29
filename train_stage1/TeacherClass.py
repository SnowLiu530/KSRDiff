#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teacher -> mask-embedding extraction -> per-class clustering -> visualization -> save prototypes
Usage example:
python teacher_prototype_pipeline.py --img_dir ./images --label_dir ./labels --out_dir ./out --train_epochs 5
"""
import os
import argparse
import pickle
from collections import defaultdict
import numpy as np
import cv2
from tqdm import tqdm
# clustering and visualization removed — this script now only trains the teacher model

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def identity_collate(batch):
    # return the single item as-is for batch_size=1
    return batch[0]

# ----------------------------
# Defaults
# ----------------------------
DEFAULT_IMG_SIZE = 128
DEFAULT_PATCH = 16
DEFAULT_BATCH = 32
DEFAULT_EMBED = 256
DEFAULT_K = 5

# ----------------------------
# Teacher model
# ----------------------------
class TeacherModel(nn.Module):
    def __init__(self, img_dim=3, mask_dim=1, embed_dim=DEFAULT_EMBED,
                 context_layers=0, context_heads=4, context_ff=512, context_dropout=0.1):
        super().__init__()
        # image patch encoder (not used heavily in this script except optional training)
        self.img_encoder = nn.Sequential(
            nn.Conv2d(img_dim, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.img_proj = nn.Linear(128, embed_dim)
        # mask patch encoder
        self.mask_encoder = nn.Sequential(
            nn.Conv2d(mask_dim, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.mask_proj = nn.Linear(64, embed_dim)

        # optional transformer-based context fusion on mask embeddings
        self.context_layers = int(context_layers)
        if self.context_layers > 0:
            # positional projection for (x,y) coords -> embed_dim
            self.pos_proj = nn.Linear(2, embed_dim)
            layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=int(context_heads), dim_feedforward=int(context_ff), dropout=float(context_dropout), batch_first=True)
            self.context_encoder = nn.TransformerEncoder(layer, num_layers=self.context_layers)
        else:
            self.pos_proj = None
            self.context_encoder = None

    def forward(self, img_patch, mask_patch):
        v = self.img_encoder(img_patch).view(img_patch.size(0), -1)
        v = self.img_proj(v)
        g = self.mask_encoder(mask_patch).view(mask_patch.size(0), -1)
        g = self.mask_proj(g)
        v = F.normalize(v, dim=-1)
        g = F.normalize(g, dim=-1)
        return v, g

# ----------------------------
# YOLOv8-seg parsing and mask utilities
# ----------------------------
# def parse_yolov8_seg_line(line):
#     # returns (cls, polygon_norm_np) or None
#     # Supports two variants:
#     # 1) Ultralytics YOLO-seg style: class cx cy w h x1 y1 x2 y2 ...
#     # 2) Polygon-only style: class x1 y1 x2 y2 ...
#     vals = line.strip().split()
#     if len(vals) < 3:
#         return None
#     try:
#         cls = int(vals[0])
#         # prefer ultralytics style if it looks like it (has bbox fields and then polygon)
#         if len(vals) >= 6 and len(vals[5:]) >= 6:
#             poly_vals = vals[5:]
#         else:
#             # fallback to polygon-only format (everything after class)
#             poly_vals = vals[1:]
#         if len(poly_vals) < 6:
#             return None
#         poly = np.array(list(map(float, poly_vals))).reshape(-1, 2)
#         return cls, poly
#     except Exception:
#         return None
def parse_yolov8_seg_line(line):
    # 使用 split() 不带参数会自动处理行尾空格和多个空格
    vals = line.strip().split()
    if len(vals) < 5:  # 一个多边形至少需要类别ID + 2个点(4个坐标)
        return None

    try:
        cls = int(vals[0])
        # 将剩余部分转为浮点数
        coords = [float(x) for x in vals[1:]]
        
        # 鲁棒性检查：如果坐标不是偶数，去掉最后一个孤立的数字
        if len(coords) % 2 != 0:
            # print(f"Warning: Odd number of coordinates for class {cls}. Truncating last value.")
            coords = coords[:-1]
            
        if len(coords) < 6: # 至少需要3个点才能构成多边形
            return None

        poly = np.array(coords, dtype=np.float32).reshape(-1, 2)
        return cls, poly
    except Exception as e:
        print(f"Error parsing line: {e}")
        return None

def polygon_to_mask_from_poly(poly_norm, H, W):
    """Convert a normalized polygon (Nx2, values in [0,1]) to a uint8 mask HxW.

    Accepts inputs as torch.Tensor, numpy array, or Python list. Returns an
    uint8 mask with foreground value 255. If input is empty/invalid returns
    a zero mask.
    """
    import torch
    # handle None
    if poly_norm is None:
        return np.zeros((H, W), dtype=np.uint8)

    # convert tensors/lists to numpy float32
    if isinstance(poly_norm, torch.Tensor):
        try:
            pts = poly_norm.detach().cpu().numpy()
        except Exception:
            pts = np.array(poly_norm)
    else:
        pts = np.array(poly_norm, dtype=np.float32)

    if pts.size == 0:
        return np.zeros((H, W), dtype=np.uint8)

    # ensure shape (N,2)
    if pts.ndim == 1:
        # flat list of coords
        if pts.size % 2 == 0:
            pts = pts.reshape(-1, 2)
        else:
            return np.zeros((H, W), dtype=np.uint8)
    elif pts.ndim == 2 and pts.shape[1] != 2:
        # try to reshape if possible
        try:
            pts = pts.reshape(-1, 2)
        except Exception:
            return np.zeros((H, W), dtype=np.uint8)

    # copy and scale to image coords
    try:
        pts = pts.astype(np.float32).copy()
        pts[:, 0] = pts[:, 0] * W
        pts[:, 1] = pts[:, 1] * H
        pts = np.round(pts).astype(np.int32)
    except Exception:
        return np.zeros((H, W), dtype=np.uint8)

    mask = np.zeros((H, W), dtype=np.uint8)
    if pts.shape[0] >= 3:
        try:
            cv2.fillPoly(mask, [pts], 255)
        except Exception:
            # if fillPoly fails for any reason, return zero mask
            return np.zeros((H, W), dtype=np.uint8)
    return mask  # uint8 0/255

# ----------------------------
# Dataset: load images and corresponding YOLO txt polygons (normalized)
# ----------------------------
class YOLOSegDataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=DEFAULT_IMG_SIZE):
        self.img_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.label_dir = label_dir
        self.img_size = img_size

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        stem = os.path.splitext(os.path.basename(img_path))[0]
        txt_path = os.path.join(self.label_dir, stem + '.txt')

        im = cv2.imread(img_path)
        if im is None:
            raise RuntimeError(f"Cannot read {img_path}")
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im_resized = cv2.resize(im, (self.img_size, self.img_size))

        instances = []
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                for line in f:
                    res = parse_yolov8_seg_line(line)
                    if res is None:
                        continue
                    cls, poly = res
                    instances.append((cls, poly))
        return im_resized, instances, img_path

# ----------------------------
# Collect mask patches per instance (sliding window non-overlapping with stride=patch_size)
# Keep only patches with mask area fraction >= min_area_ratio
# ----------------------------
def collect_mask_patches_for_image(instances, img_size, patch_size, min_area_ratio=0.02,
                                   stride=None, dilate=0, verbose=False, img_path=None,
                                   save_debug_dir=None):
    """Collect mask patches from instances.

    Args:
        instances: list of (cls, poly_norm)
        img_size: int image size (assumed square)
        patch_size: int patch size
        min_area_ratio: float ratio threshold of foreground pixels in patch
        stride: int sliding stride; if None uses patch_size (non-overlap)
        dilate: int dilation radius (in pixels) to apply to mask before sampling
        verbose: bool print per-instance stats when True
        img_path: optional path string for debug prints
        save_debug_dir: optional dir to save masks when no patches found

    Returns:
        patches: list of (cls, mask_patch_uint8, (x,y) top-left)
    """
    H = W = img_size
    if stride is None:
        stride = patch_size
    patches = []  # list of (cls, mask_patch_uint8, (x,y) top-left)
    for inst_idx, (cls, poly) in enumerate(instances):
        mask = polygon_to_mask_from_poly(poly, H, W)  # 0/255
        # optionally dilate mask to capture thin shapes
        if dilate and dilate > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate * 2 + 1, dilate * 2 + 1))
            mask = cv2.dilate(mask, k)

        total_pixels = int(np.count_nonzero(mask))
        total_area = H * W
        # sliding windows
        candidate_count = 0
        pass_count = 0
        for y in range(0, H - patch_size + 1, stride):
            for x in range(0, W - patch_size + 1, stride):
                m_patch = mask[y:y+patch_size, x:x+patch_size]
                area = int(np.count_nonzero(m_patch))
                if area > 0:
                    candidate_count += 1
                if area / (patch_size * patch_size) >= min_area_ratio:
                    patches.append((cls, m_patch.copy(), (x, y)))
                    pass_count += 1

        if verbose:
            pth = img_path if isinstance(img_path, str) else (img_path[0] if img_path else None)
            print(f"[DEBUG_INST] {pth or '<img>'} inst#{inst_idx} cls={cls} total_mask_pixels={total_pixels} \
                  candidates={candidate_count} passed={pass_count} patch_size={patch_size} min_area_ratio={min_area_ratio} stride={stride} dilate={dilate}")

        # if no patch passed and user requested saving debug masks, write mask png
        if pass_count == 0 and save_debug_dir:
            try:
                os.makedirs(save_debug_dir, exist_ok=True)
                pth = os.path.basename(img_path) if isinstance(img_path, str) else 'img'
                out_name = os.path.join(save_debug_dir, f"{pth}_inst{inst_idx}_mask.png")
                cv2.imwrite(out_name, mask)
            except Exception:
                pass
    return patches

# ----------------------------
# Extract mask embeddings using teacher.mask_encoder & mask_proj
# returns dict[class] -> list of dict items {embedding: np, mask: np.uint8, img_path, coord}
# ----------------------------
def extract_mask_embeddings(teacher, dataset, device, patch_size=DEFAULT_PATCH,
                            min_area_ratio=0.02, max_patches_per_image=200, batch_sub=64,
                            stride=None, dilate=0, verbose=False, save_debug_dir=None):
    teacher.eval()
    out = defaultdict(list)
    dl = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=identity_collate)
    with torch.no_grad():
        for im_resized, instances, img_path in tqdm(dl, desc="Extract embeddings"):
            # im_resized is numpy array (H,W,3)
            patches = collect_mask_patches_for_image(instances, dataset.img_size, patch_size,
                                                     min_area_ratio, stride=stride, dilate=dilate,
                                                     verbose=verbose, img_path=img_path,
                                                     save_debug_dir=save_debug_dir)
            if not patches:
                continue
            # cap patches per image
            if len(patches) > max_patches_per_image:
                patches = patches[:max_patches_per_image]
            mask_np_list = [p[1] for p in patches]
            mask_t = torch.from_numpy(np.stack(mask_np_list).astype(np.float32) / 255.0).unsqueeze(1).to(device)  # N x1 x ps x ps
            # compute encoder outputs for all patches of this image (in chunks) and collect
            g_list = []
            for i in range(0, mask_t.size(0), batch_sub):
                sub = mask_t[i:i+batch_sub]
                g = teacher.mask_encoder(sub).view(sub.size(0), -1)  # B x hidden
                g_list.append(g)
            if len(g_list) == 0:
                continue
            g_all = torch.cat(g_list, dim=0)  # N x hidden
            # project to embedding space
            with torch.no_grad():
                emb_all = teacher.mask_proj(g_all)  # N x D
                # optional context fusion using transformer if teacher provides it
                if getattr(teacher, 'context_encoder', None) is not None:
                    # build coords tensor (N x 2) normalized to [0,1] using dataset.img_size
                    coords = []
                    for (_cls, _mask, coord) in patches:
                        coords.append((float(coord[0]), float(coord[1])))
                    coords_t = torch.tensor(coords, dtype=torch.float32, device=emb_all.device)
                    try:
                        H = W = dataset.img_size
                        coords_t[:, 0] = coords_t[:, 0] / float(W)
                        coords_t[:, 1] = coords_t[:, 1] / float(H)
                    except Exception:
                        pass
                    pos = teacher.pos_proj(coords_t) if getattr(teacher, 'pos_proj', None) is not None else 0.0
                    emb_all = emb_all + pos
                    emb_all = teacher.context_encoder(emb_all)
                emb_all = F.normalize(emb_all, dim=-1).cpu().numpy()
            for j in range(emb_all.shape[0]):
                idx = j
                cls, mask_np, coord = patches[idx]
                out[cls].append({
                    'embedding': emb_all[j].astype(np.float32),
                    'mask': mask_np.copy(),
                    'img_path': img_path[0] if isinstance(img_path, (list, tuple)) else img_path,
                    'coord': coord
                })
    return out

# ----------------------------
# Per-class clustering into prototypes
# returns prototype_lib: dict[class] -> list of prototype dicts
# prototype dict: {'embedding': (D,), 'mask': uint8 ps x ps, 'example_img': path, 'coord': (x,y), 'count':int}
# ----------------------------
def cluster_prototypes(embeddings_per_class, n_clusters=DEFAULT_K, random_state=0):
    # clustering removed from this script. Clustering should be executed
    # as a separate step after embeddings have been extracted/saved.
    raise RuntimeError("cluster_prototypes is not available in this script. Run clustering in a separate script.")

# ----------------------------
# Visualization: t-SNE across all embeddings (colored by cls)
# ----------------------------
def visualize_tsne(embeddings_per_class, out_png, perplexity=30, random_state=0):
    raise RuntimeError("visualize_tsne removed: perform visualization in a separate analysis step")

# ----------------------------
# Save prototype libs (pkl with masks & metadata, npy for numeric arrays)
# ----------------------------
def save_prototypes(prototype_lib, out_pkl, out_npy):
    raise RuntimeError("save_prototypes removed: saving/prototyping is not part of the teacher training script")

# ----------------------------
# Make gallery pngs of prototypes (one folder per class)
# ----------------------------
def save_prototype_masks_gallery(prototype_lib, out_dir):
    raise RuntimeError("save_prototype_masks_gallery removed: prototype visualization should be done separately")

# ----------------------------
# Optional teacher training (from collected patches)
# ----------------------------
def train_teacher_on_patches(teacher, patch_list, device, batch_size=64, epochs=3, lr=1e-4, temp=0.07, contrast_weight=0.5,
                             save_example_dir=None, save_examples=16):
    # patch_list: list of (img_patch_np HWC float0..1, mask_patch_uint8)
    if not patch_list:
        print("No patches provided for teacher training — skipping training.")
        return teacher
    # optionally save a few example img/mask patches for inspection
    if save_example_dir:
        try:
            os.makedirs(save_example_dir, exist_ok=True)
            n = min(save_examples, len(patch_list))
            for i in range(n):
                img_patch, mask_np = patch_list[i]
                # img_patch: HWC float 0..1, mask_np: uint8
                try:
                    img_u8 = (img_patch * 255.0).astype('uint8')
                    # img_patch is RGB; cv2.imwrite expects BGR
                    bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(save_example_dir, f"sample_{i:03d}_img.png"), bgr)
                    cv2.imwrite(os.path.join(save_example_dir, f"sample_{i:03d}_mask.png"), mask_np)
                except Exception:
                    pass
            print(f"Saved {n} example patches to {save_example_dir}")
        except Exception:
            pass
    class PatchDataset(Dataset):
        def __init__(self, plist):
            self.plist = plist
        def __len__(self):
            return len(self.plist)
        def __getitem__(self, idx):
            img_np, mask_np = self.plist[idx]
            img_t = torch.from_numpy(img_np).permute(2,0,1).float()
            mask_t = torch.from_numpy((mask_np.astype(np.float32)/255.0)).unsqueeze(0).float()
            return img_t, mask_t
    ds = PatchDataset(patch_list)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)
    opt = torch.optim.Adam(teacher.parameters(), lr=lr)
    teacher.to(device)
    for epoch in range(epochs):
        teacher.train()
        tot = 0.0
        for img_patch, mask_patch in tqdm(dl, desc=f"Teacher train epoch {epoch+1}/{epochs}"):
            img_patch = img_patch.to(device)
            mask_patch = mask_patch.to(device)
            v, g = teacher(img_patch, mask_patch)
            loss_align = ((v - g)**2).mean()
            logits = v @ g.t() / temp
            labels = torch.arange(logits.size(0)).long().to(device)
            loss_contrast = F.cross_entropy(logits, labels)
            loss = loss_align + contrast_weight * loss_contrast
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item())
        print(f"Epoch {epoch+1} avg loss {tot/len(dl):.6f}")
    return teacher

# ----------------------------
# CLI main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", default="/mnt/share/liuxn/swinir_kg/ours(Knowledge-Guided-Diffusion)/MAR20/train/images", required=False)
    parser.add_argument("--label_dir", default="/mnt/share/liuxn/swinir_kg/ours(Knowledge-Guided-Diffusion)/MAR20/train/labels", required=False)
    parser.add_argument("--out_dir", default="/mnt/share/liuxn/swinir_kg/ours(Knowledge-Guided-Diffusion)/MAR20/prototype_out")
    parser.add_argument("--img_size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--patch_size", type=int, default=DEFAULT_PATCH)
    parser.add_argument("--embed_dim", type=int, default=DEFAULT_EMBED)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--n_clusters", type=int, default=DEFAULT_K)
    parser.add_argument("--train_epochs", type=int, default=20, help="if >0 will train teacher on collected patches")
    parser.add_argument("--teacher_ckpt", type=str, default="/mnt/share/liuxn/swinir_kg/ours(Knowledge-Guided-Diffusion)/MAR20/teacher.pth")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_area_ratio", type=float, default=0)
    parser.add_argument("--max_patches_per_image", type=int, default=200)
    parser.add_argument("--tsne_perplexity", type=int, default=30)
    parser.add_argument("--verbose", action="store_true", help="print per-image parsing/patch stats")
    parser.add_argument("--stride", type=int, default=None, help="sliding window stride (default=patch_size)")
    parser.add_argument("--dilate", type=int, default=0, help="dilate mask by this radius before sampling")
    parser.add_argument("--save_debug_dir", type=str, default=None, help="directory to save per-instance masks when no patches pass the filter")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # dataset
    ds = YOLOSegDataset(args.img_dir, args.label_dir, img_size=args.img_size)

    # teacher model
    teacher = TeacherModel(embed_dim=args.embed_dim).to(device)

    # optionally train teacher using collected positive patches
    if args.train_epochs > 0:
        # build patch list
        print("Collecting patches for teacher training...")
        patch_list = []
        dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4, collate_fn=identity_collate)
        for im_resized, instances, img_path in tqdm(dl, desc="collect patches"):
            im = im_resized if isinstance(im_resized, np.ndarray) else im_resized[0].numpy()
            save_debug_dir = args.save_debug_dir if args.save_debug_dir else (os.path.join(args.out_dir, 'debug_masks') if args.verbose else None)
            patches = collect_mask_patches_for_image(instances, args.img_size, args.patch_size, args.min_area_ratio,
                                                     stride=args.stride, dilate=args.dilate, verbose=args.verbose,
                                                     img_path=img_path, save_debug_dir=save_debug_dir)
            if args.verbose:
                pth = img_path[0] if isinstance(img_path, (list, tuple)) else img_path
                print(f"[DEBUG] {pth}: parsed_instances={len(instances)}, patches_found={len(patches)}")
            for cls, mask_np, coord in patches[:args.max_patches_per_image]:
                x, y = coord
                img_patch = im[y:y+args.patch_size, x:x+args.patch_size, :].astype(np.float32) / 255.0
                patch_list.append((img_patch, mask_np))
        print(f"Collected {len(patch_list)} patches.")

        save_example_dir = os.path.join(args.out_dir, 'sample_patches') if args.out_dir else None
        teacher = train_teacher_on_patches(teacher, patch_list, device,
                           batch_size=args.batch_size, epochs=args.train_epochs, lr=args.lr,
                           save_example_dir=save_example_dir)
        torch.save(teacher.state_dict(), args.teacher_ckpt)
        print("Saved teacher checkpoint to", args.teacher_ckpt)
    else:
        # try load checkpoint
        if os.path.exists(args.teacher_ckpt):
            sd = torch.load(args.teacher_ckpt, map_location=device)
            teacher.load_state_dict(sd)
            print("Loaded teacher from", args.teacher_ckpt)
        else:
            print("No teacher checkpoint found and train_epochs == 0. The script will still try to extract using the current (random) teacher model.")

    # This script is training-only. After training (or loading) the teacher
    # model, we exit. Embedding extraction and clustering are performed
    # in separate analysis scripts.
    print("Teacher training completed (or checkpoint loaded). Exiting.")
    return

if __name__ == "__main__":
    main()

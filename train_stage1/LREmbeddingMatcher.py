#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESSD student / paired HQ-LQ structural distillation.

Training logic:
  LQ patch --E_S + Transformer--> y_i^S --shared prototype bank--> P_i^S
  HQ patch --frozen E_img^T------> y_i^T --shared prototype bank--> P_i^T
  q_i = 1 - H(P_i^T)/log(N_e)
  L_ESSD = sum_i q_i KL(P_i^T || P_i^S) / sum_i q_i

At inference only the LQ student branch and the frozen prototype bank are used.
"""

import pickle
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchContextualizer(nn.Module):
    def __init__(self, dim, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=4 * dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, patch_embs, padding_mask=None):
        # patch_embs: [B,P,D] or [P,D]
        squeeze = patch_embs.dim() == 2
        if squeeze:
            patch_embs = patch_embs.unsqueeze(0)
        out = self.transformer(patch_embs, src_key_padding_mask=padding_mask)
        return out.squeeze(0) if squeeze else out


class PrototypeBank(nn.Module):
    """Shared fixed prototype bank used by both HQ teacher and LQ student."""

    def __init__(self, prototype_lib_path, entity_ids=None):
        super().__init__()
        with open(prototype_lib_path, "rb") as f:
            lib = pickle.load(f)

        if entity_ids is None:
            entity_ids = sorted(lib.keys(), key=lambda x: str(x))
        self.entity_ids = [str(x) for x in entity_ids]
        self.num_entities = len(self.entity_ids)

        proto_list = []
        entity_index = []
        for e, sid in enumerate(self.entity_ids):
            items = lib.get(sid, lib.get(int(sid), [])) if sid.isdigit() else lib.get(sid, [])
            if not items:
                raise ValueError(f"No prototypes found for entity {sid}.")
            for item in items:
                emb = torch.as_tensor(item["embedding"], dtype=torch.float32)
                proto_list.append(emb)
                entity_index.append(e)

        prototypes = torch.stack(proto_list, dim=0)
        prototypes = F.normalize(prototypes, dim=-1)
        self.register_buffer("prototypes", prototypes)             # [K_all,D]
        self.register_buffer(
            "prototype_entity",
            torch.tensor(entity_index, dtype=torch.long),
        )                                                           # [K_all]

    @property
    def dim(self):
        return self.prototypes.shape[-1]

    def associate(self, embeddings, temperature=0.07):
        """
        embeddings: [...,D]
        Returns:
          entity_probs [...,E]
          entity_scores [...,E]
          proto_probs [...,K_all]

        We first compute cosine similarity to all component prototypes, use the
        strongest prototype response within each entity, and then normalize
        across entities. Teacher and student use exactly the same operation.
        """
        embeddings = F.normalize(embeddings, dim=-1)
        proto = F.normalize(self.prototypes, dim=-1)
        sim = embeddings @ proto.t()                                # [...,K_all]
        proto_probs = torch.softmax(sim / temperature, dim=-1)

        scores = []
        for e in range(self.num_entities):
            mask = self.prototype_entity == e
            if not torch.any(mask):
                raise RuntimeError(f"Entity index {e} has no prototypes.")
            # robust multiple-prototype entity response
            scores.append(sim[..., mask].max(dim=-1).values)
        entity_scores = torch.stack(scores, dim=-1)                 # [...,E]
        entity_probs = torch.softmax(entity_scores / temperature, dim=-1)
        return entity_probs, entity_scores, proto_probs


class ESSDStudent(nn.Module):
    """
    LQ student with frozen HQ teacher and shared prototype bank.

    img_encoder must map [N,C,patch,patch] -> [N,D].
    teacher_encoder must map the paired HQ patches to the same D-dimensional
    structural space. The teacher is frozen here.
    """

    def __init__(
        self,
        img_encoder,
        teacher_encoder,
        prototype_lib_path,
        entity_ids=None,
        patch_size=128,
        stride=64,
        embed_dim=256,
        transformer_heads=4,
        transformer_layers=2,
        transformer_dropout=0.1,
        proto_temperature=0.07,
        device="cuda",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.patch_size = int(patch_size)
        self.stride = int(stride)
        self.proto_temperature = float(proto_temperature)

        self.student_encoder = img_encoder
        self.teacher_encoder = teacher_encoder
        self.prototype_bank = PrototypeBank(prototype_lib_path, entity_ids)

        if self.prototype_bank.dim != embed_dim:
            raise ValueError(
                f"Prototype dimension {self.prototype_bank.dim} != embed_dim {embed_dim}."
            )

        self.contextualizer = PatchContextualizer(
            embed_dim,
            num_heads=transformer_heads,
            num_layers=transformer_layers,
            dropout=transformer_dropout,
        )

        # Teacher defines the structural space and must remain fixed in ESSD.
        for p in self.teacher_encoder.parameters():
            p.requires_grad_(False)
        self.teacher_encoder.eval()

        self.to(self.device)

    @staticmethod
    def _as_bchw(img):
        """Convert numpy/tensor input to float BCHW in [0,1]."""
        if isinstance(img, np.ndarray):
            x = torch.from_numpy(img)
        elif torch.is_tensor(img):
            x = img
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")

        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            # HWC or CHW
            if x.shape[-1] in (1, 3) and x.shape[0] not in (1, 3):
                x = x.permute(2, 0, 1)
            x = x.unsqueeze(0)
        elif x.dim() == 4:
            # BHWC -> BCHW
            if x.shape[-1] in (1, 3) and x.shape[1] not in (1, 3):
                x = x.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Unsupported image shape: {tuple(x.shape)}")

        x = x.float()
        if x.numel() and x.max() > 1.5:
            x = x / 255.0
        return x.clamp(0, 1)

    def _extract_patches(self, x):
        """Aligned regular patch extraction. x is BCHW."""
        b, c, h, w = x.shape
        if h < self.patch_size or w < self.patch_size:
            new_h = max(h, self.patch_size)
            new_w = max(w, self.patch_size)
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
            h, w = new_h, new_w

        patches = F.unfold(x, kernel_size=self.patch_size, stride=self.stride)
        # [B,C*ps*ps,P] -> [B,P,C,ps,ps]
        p = patches.shape[-1]
        patches = patches.transpose(1, 2).reshape(
            b, p, c, self.patch_size, self.patch_size
        )

        coords = []
        for y in range(0, h - self.patch_size + 1, self.stride):
            for x0 in range(0, w - self.patch_size + 1, self.stride):
                coords.append((x0, y))
        return patches, coords

    def _encode_student(self, lq_patches):
        b, p, c, h, w = lq_patches.shape
        flat = lq_patches.reshape(b * p, c, h, w)
        emb = self.student_encoder(flat).reshape(b, p, -1)
        emb = F.normalize(emb, dim=-1)
        emb = self.contextualizer(emb)
        # Important: Transformer changes feature norms; normalize again before
        # cosine prototype association.
        return F.normalize(emb, dim=-1)

    @torch.no_grad()
    def _encode_teacher(self, hq_patches):
        b, p, c, h, w = hq_patches.shape
        flat = hq_patches.reshape(b * p, c, h, w)
        emb = self.teacher_encoder(flat).reshape(b, p, -1)
        return F.normalize(emb, dim=-1)

    def forward(self, lq_img, hq_img=None):
        """
        Inference:
            out = model(lq_img)
        ESSD training with paired observations:
            out = model(lq_img, hq_img)
        """
        lq = self._as_bchw(lq_img).to(self.device)
        lq_patches, coords = self._extract_patches(lq)
        y_s = self._encode_student(lq_patches)
        p_s, score_s, _ = self.prototype_bank.associate(
            y_s, temperature=self.proto_temperature
        )

        out = {
            "student_embeddings": y_s,
            "student_probs": p_s,
            "student_scores": score_s,
            "patch_coords": coords,
        }

        if hq_img is None:
            return out

        hq = self._as_bchw(hq_img).to(self.device)
        if hq.shape[-2:] != lq.shape[-2:]:
            raise ValueError(
                "ESSD requires spatially aligned HQ/LQ pairs with the same image size. "
                f"Got LQ={tuple(lq.shape[-2:])}, HQ={tuple(hq.shape[-2:])}."
            )
        hq_patches, hq_coords = self._extract_patches(hq)
        if hq_coords != coords:
            raise RuntimeError("HQ/LQ patch grids are not aligned.")

        with torch.no_grad():
            y_t = self._encode_teacher(hq_patches)
            p_t, score_t, _ = self.prototype_bank.associate(
                y_t, temperature=self.proto_temperature
            )
            confidence = self.teacher_confidence(p_t)

        out.update(
            {
                "teacher_embeddings": y_t,
                "teacher_probs": p_t,
                "teacher_scores": score_t,
                "confidence": confidence,
            }
        )
        return out

    def teacher_confidence(self, teacher_probs, eps=1e-8):
        """q_i = 1 - H(P_i^T)/log(N_e), in [0,1]."""
        e = teacher_probs.shape[-1]
        entropy = -(teacher_probs * torch.log(teacher_probs + eps)).sum(dim=-1)
        denom = torch.log(torch.tensor(float(e), device=teacher_probs.device))
        return (1.0 - entropy / (denom + eps)).clamp(0.0, 1.0)

    def essd_loss(self, outputs, eps=1e-8):
        """Uncertainty-weighted KL(P^T || P^S)."""
        if "teacher_probs" not in outputs:
            raise ValueError("Teacher outputs are required to compute ESSD loss.")
        p_s = outputs["student_probs"]
        p_t = outputs["teacher_probs"].detach()
        q = outputs["confidence"].detach()

        kl = F.kl_div(
            torch.log(p_s + eps), p_t, reduction="none"
        ).sum(dim=-1)                                                # [B,P]
        return (q * kl).sum() / (q.sum() + eps)


# -----------------------------------------------------------------------------
# Minimal training example
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    class TinyEncoder(nn.Module):
        def __init__(self, out_dim=256):
            super().__init__()
            self.output_dim = out_dim
            self.net = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.fc = nn.Linear(128, out_dim)

        def forward(self, x):
            return self.fc(self.net(x).flatten(1))

    # Usage sketch only; replace paths and load the trained HQ teacher weights.
    # student_encoder = TinyEncoder(256)
    # teacher_encoder = TinyEncoder(256)
    # teacher_encoder.load_state_dict(torch.load("teacher.pth", map_location="cpu"), strict=False)
    # model = ESSDStudent(
    #     student_encoder,
    #     teacher_encoder,
    #     "prototype_lib.pkl",
    #     entity_ids=["0", "1", "2", "3"],
    #     embed_dim=256,
    # )
    # optimizer = torch.optim.Adam(
    #     list(model.student_encoder.parameters()) + list(model.contextualizer.parameters()),
    #     lr=1e-4,
    # )
    # outputs = model(lq_batch, hq_batch)
    # loss_essd = model.essd_loss(outputs)
    # optimizer.zero_grad(set_to_none=True)
    # loss_essd.backward()
    # optimizer.step()
    pass

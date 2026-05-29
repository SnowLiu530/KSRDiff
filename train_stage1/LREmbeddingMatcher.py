import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms, models
import pickle
from tqdm import tqdm

import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
import pickle
from tqdm import tqdm


# ============================
# 1) Patch Transformer 上下文化模块
# ============================

class PatchContextualizer(nn.Module):
    """
    对 patch-level embedding 进行上下文化建模：
    输入: (N_patch, D)
    输出: (N_patch, D)
    """
    def __init__(self, dim, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, patch_embs):
        """
        patch_embs: (N, D)
        返回上下文化后的 patch_embs
        """
        x = patch_embs.unsqueeze(0)        # (1, N, D)
        x = self.transformer(x)            # (1, N, D)
        return x.squeeze(0)                # (N, D)


# ============================
# 2) 主类：LREmbeddingMatcher（加入 Transformer）
# ============================

class LREmbeddingMatcher(nn.Module):
    def __init__(self, img_encoder, prototype_lib_path, device='cuda',
                 patch_size=128, stride=64, teacher_encoder=None,
                 sid_list=None,
                 use_transformer=True,          # ★ 新增参数
                 transformer_dim=256,
                 transformer_heads=4,
                 transformer_layers=2):

        super().__init__()
        self.device = device
        self.img_encoder = img_encoder.to(device)
        self.img_encoder.train()
        self.teacher_encoder = teacher_encoder.to(device).eval() if teacher_encoder else None

        self.patch_size = patch_size
        self.stride = stride

        # ------------------- 原型库 -------------------
        with open(prototype_lib_path, 'rb') as f:
            self.prototype_lib = pickle.load(f)

        if sid_list is not None:
            self.sid_list = list(sid_list)
        else:
            self.sid_list = sorted(list(self.prototype_lib.keys()), key=lambda x: str(x))

        # build prototype embeddings (num_proto x D)
        self.prototype_embeddings = {}
        self.prototype_counts = {}
        self.prototype_centroids = {}

        for sid in self.sid_list:
            prototypes = self.prototype_lib.get(sid, [])
            emb_arrays = []
            centroids = []
            for p in prototypes:
                try:
                    e = np.asarray(p["embedding"], dtype=np.float32)
                    if e.ndim == 1:
                        emb_arrays.append(e)
                except:
                    continue

                c = p.get("centroid", None)
                if c is not None:
                    try:
                        centroids.append((float(c[0]), float(c[1])))
                    except:
                        pass

            if len(emb_arrays) == 0:
                self.prototype_embeddings[sid] = None
                self.prototype_counts[sid] = 0
                self.prototype_centroids[sid] = None
                continue

            proto_mat = torch.from_numpy(np.stack(emb_arrays)).to(self.device)
            proto_mat = F.normalize(proto_mat, p=2, dim=1)
            self.prototype_embeddings[sid] = proto_mat
            self.prototype_counts[sid] = proto_mat.shape[0]

            if len(centroids) > 0:
                xs = [c[0] for c in centroids]
                ys = [c[1] for c in centroids]
                self.prototype_centroids[sid] = (float(np.mean(xs)), float(np.mean(ys)))
            else:
                self.prototype_centroids[sid] = None

        # ------------------- 图像归一化 -------------------
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        # ------------------- 上下文化 Transformer（可选） -------------------
        self.use_transformer = use_transformer
        if use_transformer:
            self.contextualizer = PatchContextualizer(
                dim=transformer_dim,
                num_heads=transformer_heads,
                num_layers=transformer_layers
            ).to(device)

        

    # ------------------- Patch 提取 -------------------
    def extract_patches(self, img):
        # img expected H x W x C (numpy) or similar
        # Defensive handling:
        # - accept torch.Tensor (CPU/GPU), numpy arrays
        # - accept batch tensors (B,C,H,W) or (B,H,W,C) -> use first sample
        # - accept CHW (C,H,W) and convert to HWC
        # - ensure numeric range is suitable for downstream transforms
        # Normalize to a numpy H x W x C array (uint8 or float in [0,1])
        arr = None
        # 1) If torch tensor, move to cpu and convert
        if isinstance(img, torch.Tensor):
            try:
                arr = img.detach().cpu().numpy()
            except Exception:
                # fallback to copying to cpu first
                arr = img.detach().to('cpu').numpy()
        else:
            arr = np.asarray(img)

        # 2) Handle batch dimension
        if arr.ndim == 4:
            # could be (B, C, H, W) or (B, H, W, C) -> pick first sample
            if arr.shape[1] in (1, 3):
                # assume BCHW
                arr = arr[0]
            else:
                # assume BHWC
                arr = arr[0]

        # 3) Now arr should be HWC or CHW or HW
        if arr.ndim == 3:
            # CHW -> HWC
            if arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
                arr = np.transpose(arr, (1, 2, 0))
        elif arr.ndim == 2:
            # HW -> HWC
            arr = arr[:, :, None]
        else:
            raise ValueError(f'extract_patches: unsupported img shape {arr.shape}')

        H, W, C = arr.shape
        img = arr

        # 4) Heuristic: if float image in normalized range, convert to uint8 0-255
        try:
            if np.issubdtype(img.dtype, np.floating):
                mx = float(np.nanmax(img))
                mn = float(np.nanmin(img))
                # common normalizations: [-0.5,0.5] or [0,1]
                if mn >= -1.0 and mx <= 1.0:
                    if mn < 0.0:
                        # assume centered at 0 (e.g., [-0.5,0.5])
                        img = ((img + 0.5) * 255.0).clip(0, 255).astype('uint8')
                    else:
                        # assume [0,1]
                        img = (img * 255.0).clip(0, 255).astype('uint8')
                else:
                    # if values already in 0-255 but float, cast
                    img = img.clip(0, 255).astype('uint8')
        except Exception:
            # if anything goes wrong, keep original
            pass

        # If image is smaller than patch_size, resize it up to patch_size to guarantee at least one patch
        if H < self.patch_size or W < self.patch_size:
            try:
                import cv2
                img = cv2.resize(img, (self.patch_size, self.patch_size))
                H, W, C = img.shape
            except Exception:
                # fallback: pad with zeros
                new_img = np.zeros((self.patch_size, self.patch_size, C), dtype=img.dtype)
                new_img[0:H, 0:W, ...] = img
                img = new_img
                H, W, C = img.shape

        patches = []
        for y in range(0, H - self.patch_size + 1, self.stride):
            for x in range(0, W - self.patch_size + 1, self.stride):
                patch = img[y:y + self.patch_size, x:x + self.patch_size]
                patches.append((patch, (x, y)))
        return patches

    # ------------------- 原型匹配 -------------------
    def match_prototypes(self, patch_emb):
        P = patch_emb.size(0)
        E = len(self.sid_list)
        device = patch_emb.device

        raw_scores = torch.zeros((P, E), device=device, dtype=patch_emb.dtype)
        for e, sid in enumerate(self.sid_list):
            proto_tensor = self.prototype_embeddings.get(sid, None)
            if proto_tensor is None:
                raw_scores[:, e] = 0.0
                continue
            sim = torch.matmul(patch_emb, proto_tensor.t())  # (P x num_proto)
            sid_score, _ = sim.max(dim=1)
            raw_scores[:, e] = sid_score

        if raw_scores.numel() == 0 or raw_scores.size(1) == 0:
            entity_probs = torch.zeros((P, 0), device=device)
        else:
            entity_probs = torch.softmax(raw_scores, dim=1)

        return entity_probs, raw_scores

    # ------------------- Forward（加入 Transformer） -------------------
    def forward(self, lr_img):
        # Support both single-image inputs and batched inputs (Tensor BxCxHxW)
        is_tensor = isinstance(lr_img, torch.Tensor)
        if is_tensor and lr_img.dim() == 4:
            B = lr_img.shape[0]
            per_image_patch_embs = []
            per_image_patch_coords = []
            per_image_entity_probs = []
            per_image_teacher_embs = []

            for b in range(B):
                sample = lr_img[b]
                patches = self.extract_patches(sample)
                coords = []
                # batch encode patches for this image
                if len(patches) == 0:
                    # fallback: empty outputs
                    per_image_patch_embs.append(torch.zeros((0, getattr(self.img_encoder, 'output_dim', 128)), device=self.device))
                    per_image_entity_probs.append(torch.zeros((0, len(self.sid_list)), device=self.device))
                    per_image_patch_coords.append([])
                    if self.teacher_encoder:
                        per_image_teacher_embs.append(torch.zeros((0, getattr(self.img_encoder, 'output_dim', 128)), device=self.device))
                    continue

                patch_tensors = []
                for patch, (x, y) in patches:
                    coords.append((x, y))
                    pt = self.transform(patch)
                    patch_tensors.append(pt)
                patch_tensors = torch.stack(patch_tensors, dim=0).to(self.device)

                # 确保特征提取主干在 eval 模式，避免 BN 在小批量/单补丁时报错
                try:
                    self.img_encoder.eval()
                except Exception:
                    pass
                with torch.no_grad() if self.img_encoder is None else torch.enable_grad():
                    patch_emb = self.img_encoder(patch_tensors)
                patch_emb = F.normalize(patch_emb, p=2, dim=1)

                if self.use_transformer:
                    patch_emb = self.contextualizer(patch_emb)

                entity_probs, raw_scores = self.match_prototypes(patch_emb)

                per_image_patch_embs.append(patch_emb)
                per_image_entity_probs.append(entity_probs)
                per_image_patch_coords.append(coords)

                if self.teacher_encoder:
                    with torch.no_grad():
                        t_emb = self.teacher_encoder(patch_tensors)
                        t_emb = F.normalize(t_emb, p=2, dim=1)
                    per_image_teacher_embs.append(t_emb)

            # stack per-image results into batch tensors
            # assume all images have same number of patches L
            Ls = [x.shape[0] for x in per_image_patch_embs]
            if len(set(Ls)) != 1:
                # if patch counts vary, pad with zeros to the max L
                maxL = max(Ls)
                D = per_image_patch_embs[0].shape[1] if Ls[0] > 0 else getattr(self.img_encoder, 'output_dim', 128)
                padded_embs = []
                padded_entities = []
                padded_teachers = []
                for i in range(B):
                    Li = per_image_patch_embs[i].shape[0]
                    if Li < maxL:
                        pad = torch.zeros((maxL - Li, D), device=self.device)
                        padded = torch.cat([per_image_patch_embs[i], pad], dim=0)
                    else:
                        padded = per_image_patch_embs[i]
                    padded_embs.append(padded)

                    Ei = per_image_entity_probs[i].shape[0]
                    if Ei < maxL:
                        pad_e = torch.zeros((maxL - Ei, per_image_entity_probs[i].shape[1]), device=self.device)
                        padded_e = torch.cat([per_image_entity_probs[i], pad_e], dim=0)
                    else:
                        padded_e = per_image_entity_probs[i]
                    padded_entities.append(padded_e)

                    if self.teacher_encoder:
                        Ti = per_image_teacher_embs[i].shape[0]
                        if Ti < maxL:
                            pad_t = torch.zeros((maxL - Ti, D), device=self.device)
                            padded_t = torch.cat([per_image_teacher_embs[i], pad_t], dim=0)
                        else:
                            padded_t = per_image_teacher_embs[i]
                        padded_teachers.append(padded_t)

                batch_patch_embs = torch.stack(padded_embs, dim=0)
                batch_entity_probs = torch.stack(padded_entities, dim=0)
                batch_patch_coords = per_image_patch_coords
                if self.teacher_encoder:
                    batch_teacher_embs = torch.stack(padded_teachers, dim=0)
            else:
                batch_patch_embs = torch.stack(per_image_patch_embs, dim=0)
                batch_entity_probs = torch.stack(per_image_entity_probs, dim=0)
                batch_patch_coords = per_image_patch_coords
                if self.teacher_encoder:
                    batch_teacher_embs = torch.stack(per_image_teacher_embs, dim=0)

            if self.teacher_encoder:
                return batch_patch_embs, batch_patch_coords, batch_entity_probs, batch_teacher_embs
            else:
                return batch_patch_embs, batch_patch_coords, batch_entity_probs

        # Single-image path (existing behavior)
        patches = self.extract_patches(lr_img)

        patch_coords = []
        patch_embs = []
        teacher_embs = []

        # 1) 编码每个 patch
        for patch, (x, y) in patches:
            patch_tensor = self.transform(patch).unsqueeze(0).to(self.device)
            try:
                self.img_encoder.eval()
            except Exception:
                pass
            patch_emb = self.img_encoder(patch_tensor)   # (1, D)
            patch_emb = F.normalize(patch_emb, p=2, dim=1)

            patch_embs.append(patch_emb)
            patch_coords.append((x, y))

            if self.teacher_encoder:
                with torch.no_grad():
                    t_emb = F.normalize(self.teacher_encoder(patch_tensor), p=2, dim=1)
                teacher_embs.append(t_emb)

        patch_embs = torch.cat(patch_embs, dim=0)  # (N, D)

        # 2) Transformer 上下文化（最重要步骤）
        if self.use_transformer:
            patch_embs = self.contextualizer(patch_embs)

        # 3) 匹配原型库
        entity_probs, raw_scores = self.match_prototypes(patch_embs)

        # 4) 返回 — 保持返回格式兼容
        if self.teacher_encoder:
            teacher_embs = torch.cat(teacher_embs, dim=0)
            return patch_embs, patch_coords, entity_probs, teacher_embs
        else:
            return patch_embs, patch_coords, entity_probs


"""
===================== 使用/训练示例 =====================

# 1) 初始化 LR encoder + optional HR teacher

lr_encoder = MyImageEncoder()
hr_encoder = MyHRImageEncoder()  # 预训练权重
matcher = LREmbeddingMatcher(lr_encoder, prototype_lib_path='prototype_lib.pkl',
patch_size=128, stride=64, teacher_encoder=hr_encoder)

# 2) 定义优化器

optimizer = torch.optim.Adam(lr_encoder.parameters(), lr=1e-4)

# 3) 训练循环（蒸馏）

for epoch in range(num_epochs):
for lr_img in dataloader:  # 逐图/逐 batch
optimizer.zero_grad()
patch_embs, teacher_embs, patch_coords = matcher(lr_img)
# ----------------- 蒸馏 loss -----------------
# L2 loss: LR embedding 对齐 HR embedding
loss = F.mse_loss(patch_embs, teacher_embs)
# 或者 CLIP-style contrastive loss，patch_embs 与 teacher_embs 为正样本
loss.backward()
optimizer.step()

# 4) 推理/可视化

# 调用示例（注意 match_prototypes 总是返回 (entity_probs, raw_scores)）
patch_embs, patch_coords = matcher.forward(lr_img)
entity_probs, raw_scores = matcher.match_prototypes(patch_embs)

# 后续可生成 pixel-smoothed 或 patch-block 可视化

========================================================
"""



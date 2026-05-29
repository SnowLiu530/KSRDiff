#!/usr/bin/env python3
import os
import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import torch
import torch.nn as nn

# -------------------- 配置 --------------------
txt_dir = '/mnt/share/liuxn/swinir_kg/ours(Knowledge-Guided-Diffusion)/dataset_r/train/labels'  # 输入 txt 标注
out_dir = '/mnt/share/liuxn/swinir_kg/ours(Knowledge-Guided-Diffusion)/dataset_r/prototype_lib'
vis_root = '/mnt/share/liuxn/swinir_kg/ours(Knowledge-Guided-Diffusion)/dataset_r/prototype_vis'
# out_dir = './prototype_lib'
# vis_root = './prototype_vis'
os.makedirs(out_dir, exist_ok=True)
os.makedirs(vis_root, exist_ok=True)

encoding_dim = 16  # poly编码向量维度

# -------------------- 函数 --------------------
def get_parametric_template(pts, n_vertices=4):
    """
    将 pts 转化为 n_vertices 多边形模板，并归一化质心到原点，缩放到单位范围
    返回 (poly_norm, centroid)
    """
    hull = cv2.convexHull(pts.astype(np.float32))
    hull_pts = hull.reshape(-1, 2)
    n_hull = hull_pts.shape[0]

    if n_hull >= n_vertices:
        idxs = np.linspace(0, n_hull - 1, n_vertices, dtype=int)
        poly = hull_pts[idxs]
    else:
        # 不足 n_vertices 使用最小外接矩形插值
        rect = cv2.minAreaRect(pts)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype=np.float32)
        pts_list = [p for p in box]
        insert_pos = 0
        while len(pts_list) < n_vertices:
            a = np.array(pts_list[insert_pos % len(pts_list)])
            b = np.array(pts_list[(insert_pos + 1) % len(pts_list)])
            mid = ((a + b)/2.0).tolist()
            pts_list.insert(insert_pos + 1, mid)
            insert_pos += 2
        poly = np.array(pts_list[:n_vertices], dtype=np.float32)

    # 质心归一化
    centroid = poly.mean(axis=0)
    poly -= centroid
    # 归一化最大半径为1
    scale = np.linalg.norm(poly, axis=1).max()
    if scale > 0:
        poly /= scale
    return poly, centroid

# -------------------- Poly编码网络 --------------------
class PolyEncoder(nn.Module):
    def __init__(self, n_vertices, encoding_dim=16):
        super().__init__()
        self.n_vertices = n_vertices
        self.fc = nn.Sequential(
            nn.Linear(n_vertices*2, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim)
        )
    def forward(self, poly):
        # poly shape: [n_vertices, 2]
        x = poly.flatten().float()
        return self.fc(x)

# 分别为 4 顶点和 5 顶点准备编码器
poly_encoder4 = PolyEncoder(n_vertices=4, encoding_dim=encoding_dim)
poly_encoder5 = PolyEncoder(n_vertices=5, encoding_dim=encoding_dim)
poly_encoder4.eval()
poly_encoder5.eval()

# -------------------- 读取 txt 并生成模板 --------------------
print("开始从 TXT 标注生成参数化模板...")
templates_poly = defaultdict(list)  # sid -> list of poly
templates_feat = defaultdict(list)  # sid -> list of encoding vector
prototype_lib = defaultdict(list)   # sid -> list of dicts for training

txt_files = sorted([f for f in os.listdir(txt_dir) if f.endswith('.txt')])
for txt_file in tqdm(txt_files, desc="Processing TXT files"):
    path = os.path.join(txt_dir, txt_file)
    with open(path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        sid = int(parts[0])
        coords = np.array([float(x) for x in parts[1:]])
        if len(coords) >= 6 and len(coords) % 2 == 0:
            pts = coords.reshape(-1, 2)
        elif len(coords) == 4:  # bbox: cx, cy, w, h
            cx, cy, bw, bh = coords
            x1 = cx - bw/2
            y1 = cy - bh/2
            x2 = cx + bw/2
            y2 = cy + bh/2
            pts = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])
        else:
            continue

        pts = pts.astype(np.float32)
        n_vertices = 4 if sid in [0,1] else 5
        poly, centroid = get_parametric_template(pts, n_vertices=n_vertices)

        # 去重
        exist = False
        for t in templates_poly[sid]:
            if np.allclose(poly, t, atol=1e-2):
                exist = True
                break
        if not exist:
            templates_poly[sid].append(poly)
            # 编码并存储
            poly_tensor = torch.tensor(poly)
            encoder = poly_encoder4 if n_vertices == 4 else poly_encoder5
            with torch.no_grad():
                feat = encoder(poly_tensor)
            feat_np = feat.numpy()
            templates_feat[sid].append(feat_np)
            prototype_lib[sid].append({
                'embedding': feat_np,
                'mask_embedding': feat_np,
                'poly': poly,
                'centroid': centroid
            })

# -------------------- 保存 --------------------
with open(os.path.join(out_dir, 'prototypes_poly.pkl'), 'wb') as f:
    pickle.dump(templates_poly, f)
with open(os.path.join(out_dir, 'prototypes_feat.pkl'), 'wb') as f:
    pickle.dump(templates_feat, f)
with open(os.path.join(out_dir, 'prototype_lib.pkl'), 'wb') as f:
    pickle.dump(prototype_lib, f)
print("参数化模板及编码保存完成:", out_dir)

# -------------------- 可视化模板 --------------------
print("\n开始可视化生成的模板...")
for sid, poly_list in tqdm(templates_poly.items(), desc="Visualizing templates"):
    sid_dir = os.path.join(vis_root, f'id_{sid}')
    os.makedirs(sid_dir, exist_ok=True)
    for idx, poly in enumerate(poly_list):
        plt.figure(figsize=(4,4))
        plt.title(f'Semantic ID {sid} - Template {idx}')
        plt.axis('equal')
        plt.xlim([-1.1,1.1])
        plt.ylim([-1.1,1.1])
        plt.grid(True, linestyle='--', alpha=0.3)

        # 绘制闭合多边形
        poly_closed = np.vstack([poly, poly[0]])
        plt.plot(poly_closed[:,0], poly_closed[:,1], 'b-o', markersize=5)
        plt.scatter(0,0,c='red', label='centroid')
        plt.legend()
        save_path = os.path.join(sid_dir, f'template_{idx}.png')
        plt.savefig(save_path)
        plt.close()
    print(f"Semantic ID {sid}: {len(poly_list)} templates visualized in {sid_dir}")

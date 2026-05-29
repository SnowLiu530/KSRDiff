#!/usr/bin/env python3
"""
CLIP-style 三元对齐 pipeline + KMeans 聚类去冗余 + embedding 保存
Resume 方案 1: 只加载模型权重，不加载优化器状态
"""
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.cluster import KMeans

# -------------------- 配置 --------------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-4
EMBED_DIM = 128
IMG_SIZE = 128
TEMP = 0.07  # InfoNCE 温度
EARLY_STOP_PATIENCE = 10
N_CLUSTERS = 8  # 每个 semantic_id 的聚类中心数

# IMG_DIR = './dataset/train/images'
# LABEL_DIR = './dataset/train/labels'
# CHECKPOINT_DIR = './checkpoints'
# PROTOTYPE_LIB_DIR = './prototype_lib'
# PROTOTYPE_VIS_DIR = './prototype_vis'
IMG_DIR = '/mnt/share/liuxn/swinir_kg/ours(Knowledge-Guided-Diffusion)/MAR20/train/images'
LABEL_DIR = '/mnt/share/liuxn/swinir_kg/ours(Knowledge-Guided-Diffusion)/MAR20/train/labels'
CHECKPOINT_DIR = '/mnt/share/liuxn/swinir_kg/ours(Knowledge-Guided-Diffusion)/MAR20/best_encoders'
PROTOTYPE_LIB_DIR = '/mnt/share/liuxn/swinir_kg/ours(Knowledge-Guided-Diffusion)/MAR20/prototype_lib'
PROTOTYPE_VIS_DIR = '/mnt/share/liuxn/swinir_kg/ours(Knowledge-Guided-Diffusion)/MAR20/prototype_vis'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(PROTOTYPE_LIB_DIR, exist_ok=True)
os.makedirs(PROTOTYPE_VIS_DIR, exist_ok=True)

RESUME = True   # 是否从 checkpoint 加载模型

# -------------------- Dataset --------------------
class PartAlignDataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=IMG_SIZE):
        self.img_paths = sorted([os.path.join(img_dir,f) for f in os.listdir(img_dir) if f.endswith(('.png','.jpg'))])
        self.label_paths = sorted([os.path.join(label_dir,f) for f in os.listdir(label_dir) if f.endswith('.txt')])
        self.img_size = img_size
        self.samples = []
        for img_path, label_path in zip(self.img_paths, self.label_paths):
            img = cv2.imread(img_path)
            H, W = img.shape[:2]
            with open(label_path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if not parts: continue
                sid = int(parts[0])
                coords = np.array([float(x) for x in parts[1:]])
                if len(coords) >= 6 and len(coords)%2==0:
                    pts = coords.reshape(-1,2)
                elif len(coords)==4:
                    cx,cy,w,h = coords
                    x1,y1 = cx-w/2, cy-h/2
                    x2,y2 = cx+w/2, cy+h/2
                    pts = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])
                else:
                    continue
                mask = np.zeros((H,W), dtype=np.uint8)
                cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
                x,y,w_box,h_box = cv2.boundingRect(pts.astype(np.int32))
                img_crop = cv2.resize(img[y:y+h_box, x:x+w_box], (img_size,img_size))
                mask_crop = cv2.resize(mask[y:y+h_box, x:x+w_box], (img_size,img_size), interpolation=cv2.INTER_NEAREST)
                # 多边形采样
                def sample_poly(pts, n_vertices=5):
                    pts = pts.astype(np.float32)
                    hull = cv2.convexHull(pts)
                    hull_pts = hull.reshape(-1, 2)
                    n_hull = hull_pts.shape[0]
                    if n_hull >= n_vertices:
                        idxs = np.linspace(0, n_hull-1, n_vertices, dtype=int)
                        poly = hull_pts[idxs]
                    else:
                        rect = cv2.minAreaRect(pts)
                        box = cv2.boxPoints(rect)
                        box = np.array(box, dtype=np.float32)
                        pts_list = [p for p in box]
                        insert_pos=0
                        while len(pts_list)<n_vertices:
                            a=np.array(pts_list[insert_pos % len(pts_list)])
                            b=np.array(pts_list[(insert_pos+1)%len(pts_list)])
                            mid=((a+b)/2).tolist()
                            pts_list.insert(insert_pos+1, mid)
                            insert_pos+=2
                        poly = np.array(pts_list[:n_vertices], dtype=np.float32)
                    return poly
                poly = sample_poly(pts, n_vertices=5)
                centroid = poly.mean(axis=0)
                attr = np.concatenate([poly.flatten(), centroid])  # 10+2=12
                self.samples.append((img_crop, mask_crop, attr, sid, poly, centroid))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img, mask, attr, sid, poly, centroid = self.samples[idx]
        img = torch.tensor(img.transpose(2,0,1), dtype=torch.float32)/255.0
        mask = torch.tensor(mask[None], dtype=torch.float32)
        attr = torch.tensor(attr, dtype=torch.float32)
        return img, mask, attr, sid, poly, centroid

def custom_collate(batch):
    imgs, masks, attrs, sids, polys, centroids = zip(*batch)
    return torch.stack(imgs), torch.stack(masks), torch.stack(attrs), sids, polys, centroids

# -------------------- Encoders --------------------
class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, embed_dim)
    def forward(self, x):
        return self.model(x)

class MaskEncoder(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(64, embed_dim)
    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class PolyEncoder(nn.Module):
    def __init__(self, input_dim=12, embed_dim=EMBED_DIM):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, embed_dim)
        )
    def forward(self, x):
        return self.fc(x)

# -------------------- DataLoader --------------------
if __name__ == '__main__':
    dataset = PartAlignDataset(IMG_DIR, LABEL_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate)

    # -------------------- Model & Optimizer --------------------
    img_encoder = ImageEncoder().to(DEVICE)
    mask_encoder = MaskEncoder().to(DEVICE)
    poly_encoder = PolyEncoder().to(DEVICE)
    params = list(img_encoder.parameters()) + list(mask_encoder.parameters()) + list(poly_encoder.parameters())
    optimizer = torch.optim.Adam(params, lr=LR)

    # -------------------- Resume --------------------
    if RESUME:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, 'best_encoders.pth')
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            img_encoder.load_state_dict(checkpoint['img_encoder'])
            mask_encoder.load_state_dict(checkpoint['mask_encoder'])
            poly_encoder.load_state_dict(checkpoint['poly_encoder'])
            print("Model weights loaded. Optimizer not loaded (will start fresh).")

    # -------------------- Training --------------------
    print("Start training with CLIP-style 3-way alignment...")
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(EPOCHS):
        img_encoder.train(); mask_encoder.train(); poly_encoder.train()
        total_loss = 0
        for img, mask, attr, sids, polys, centroids in tqdm(dataloader):
            img = img.to(DEVICE); mask = mask.to(DEVICE); attr = attr.to(DEVICE)
            img_emb = F.normalize(img_encoder(img), dim=1)
            mask_emb = F.normalize(mask_encoder(mask), dim=1)
            poly_emb = F.normalize(poly_encoder(attr), dim=1)

            logits_img_mask = torch.matmul(img_emb, mask_emb.T)/TEMP
            logits_img_poly = torch.matmul(img_emb, poly_emb.T)/TEMP
            logits_mask_poly = torch.matmul(mask_emb, poly_emb.T)/TEMP
            labels = torch.arange(img_emb.size(0), device=DEVICE)
            loss = (F.cross_entropy(logits_img_mask, labels) +
                    F.cross_entropy(logits_img_poly, labels) +
                    F.cross_entropy(logits_mask_poly, labels))

            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        # 早停 & 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save({
                'img_encoder': img_encoder.state_dict(),
                'mask_encoder': mask_encoder.state_dict(),
                'poly_encoder': poly_encoder.state_dict()
            }, os.path.join(CHECKPOINT_DIR, 'best_encoders.pth'))
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # -------------------- 原型库生成 & 聚类 --------------------
    print("Generating prototype library with KMeans clustering...")
    prototype_lib = defaultdict(list)
    img_encoder.eval(); mask_encoder.eval(); poly_encoder.eval()
    all_embeddings = defaultdict(list)
    all_polys = defaultdict(list)
    all_centroids = defaultdict(list)
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            img, mask, attr, sid, poly, centroid = dataset[i]
            attr_tensor = attr.unsqueeze(0).to(DEVICE)
            emb = F.normalize(poly_encoder(attr_tensor), dim=1).squeeze(0).cpu().numpy()
            all_embeddings[sid].append(emb)
            all_polys[sid].append(poly)
            all_centroids[sid].append(centroid)

    # 聚类
    for sid in all_embeddings:
        X = np.stack(all_embeddings[sid])
        kmeans = KMeans(n_clusters=min(N_CLUSTERS, len(X)), random_state=0).fit(X)
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        for c_idx, center in enumerate(centers):
            idx = np.where(labels==c_idx)[0][0]
            prototype_lib[sid].append({
                'embedding': center,
                'poly': all_polys[sid][idx],
                'centroid': all_centroids[sid][idx]
            })

    # 保存原型库
    with open(os.path.join(PROTOTYPE_LIB_DIR,'prototype_lib.pkl'),'wb') as f:
        pickle.dump(prototype_lib,f)
    print(f"Prototype library saved to {PROTOTYPE_LIB_DIR}/prototype_lib.pkl")

    # 可视化
    def visualize_template(poly, centroid, save_path):
        plt.figure(figsize=(4,4))
        plt.axis('equal')
        plt.xlim([-1.1,1.1]); plt.ylim([-1.1,1.1])
        plt.grid(True, linestyle='--', alpha=0.3)
        poly_closed = np.vstack([poly, poly[0]])
        plt.plot(poly_closed[:,0], poly_closed[:,1],'b-o',markersize=5)
        plt.scatter(centroid[0],centroid[1],c='red',label='centroid')
        plt.legend(); plt.savefig(save_path); plt.close()

    for sid, tpl_list in prototype_lib.items():
        sid_dir = os.path.join(PROTOTYPE_VIS_DIR,f'id_{sid}')
    os.makedirs(sid_dir, exist_ok=True)
    for idx, tpl in enumerate(tpl_list):
        poly = tpl['poly'] - tpl['centroid']
        scale = np.linalg.norm(poly,axis=1).max()
        if scale>0: poly/=scale
        visualize_template(poly,[0,0],os.path.join(sid_dir,f'template_{idx}.png'))
print(f"Prototype visualization saved to {PROTOTYPE_VIS_DIR}/")

# -------------------- 使用说明 --------------------
print("\nTraining finished. You can load the encoders and prototype library as follows:\n")
print("""
# Load encoders
checkpoint = torch.load('./checkpoints/best_encoders.pth', map_location=DEVICE)
img_encoder.load_state_dict(checkpoint['img_encoder'])
mask_encoder.load_state_dict(checkpoint['mask_encoder'])
poly_encoder.load_state_dict(checkpoint['poly_encoder'])
img_encoder.eval(); mask_encoder.eval(); poly_encoder.eval()

# Load prototype library
with open('./prototype_lib/prototype_lib.pkl','rb') as f:
    prototype_lib = pickle.load(f)

# Now you can use poly_encoder(attr) to get embedding for LR query,
# then match against prototype_lib embeddings (e.g., cosine similarity)
""")

#!/usr/bin/env python3
"""
Prototype / patch embedding visualization & graph view for first N test images.
- t-SNE / UMAP of prototype embeddings vs student patch embeddings (and optional teacher).
- Prototype matching heatmaps (student / teacher) via top-K similarity scatter to spatial map.
- GNN (knowledge graph) visualization with edge weights (initial vs refined).

Usage example:
python tools/vis_proto_analysis.py \
  --data_path /mnt/share/liuxn/swinir_kg/aircraft_babiao/dataset \
  --ckpt /mnt/share/liuxn/swinir_kg/ours(Knowledge-Guided-Diffusion)/experiments/exp_aircraft_babiao/last_kgda.pth \
  --prototype_lib_path /mnt/share/liuxn/swinir_kg/aircraft_babiao/prototype_lib/prototypes.pkl \
  --knowledge_graph_path /mnt/share/liuxn/swinir_kg/kg.json \
  --teacher_ckpt /mnt/share/liuxn/swinir_kg/aircraft_babiao/teacher.pth \
  --output_dir runs/vis_debug --num_samples 5 --topk 3 --reduce umap

This script does NOT modify training/inference code; it only imports helpers.
"""
import argparse
import copy
import json
import os
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE

# ensure we can import project-level modules when running from tools/
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import umap
    _HAS_UMAP = True
except Exception:
    _HAS_UMAP = False

try:
    import pyiqa  # optional; only used for potential metric hooks
except Exception:
    pyiqa = None

# Local imports (no modification to existing files)
from infer_sr import (
    LRDataset,
    build_model_from_ckpt,
    build_condition_modules,
    build_cond_map_from_modules,
    scatter_scores_to_map,
)
from train_stage1.TeacherClass import TeacherModel  # teacher encoder


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def load_teacher(ckpt_path, device, embed_dim=None):
    if ckpt_path is None or (not os.path.exists(ckpt_path)):
        return None
    state = torch.load(ckpt_path, map_location=device)
    # Try to infer embed_dim from checkpoint weights (e.g., img_proj.weight)
    inferred_dim = None
    for k, v in state.items():
        if k.endswith('img_proj.weight') and v.dim() == 2:
            inferred_dim = v.shape[0]
            break
    for k, v in state.items():
        if inferred_dim is None and k.endswith('mask_proj.weight') and v.dim() == 2:
            inferred_dim = v.shape[0]
            break
    embed_dim = embed_dim or inferred_dim or 128
    teacher = TeacherModel(img_dim=3, mask_dim=1, embed_dim=embed_dim).to(device)
    # Robust load: filter matching shapes only
    model_sd = teacher.state_dict()
    filtered = {}
    mismatched = []
    for k, v in state.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            filtered[k] = v
        elif k in model_sd:
            mismatched.append((k, v.shape, model_sd[k].shape))
    if mismatched:
        warnings.warn(f"Teacher load: drop mismatched keys: {mismatched[:5]}" + ("..." if len(mismatched)>5 else ""))
    missing, unexpected = teacher.load_state_dict(filtered, strict=False)
    if missing or unexpected:
        warnings.warn(f"Teacher load with missing={missing}, unexpected={unexpected}")
    teacher.eval()
    return teacher


def reduce_embeddings(emb_list, labels, method="tsne", random_state=0):
    """Return 2D coords for embeddings; emb_list is list of arrays (N_i, D)."""
    if len(emb_list) == 0:
        return np.zeros((0, 2)), np.zeros((0,))
    X = np.concatenate(emb_list, axis=0)
    label_arr = np.concatenate(labels, axis=0)
    n = X.shape[0]
    if n < 2:
        return X, label_arr
    # guard NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if n < 2:
        return X, label_arr
    if method == "umap" and _HAS_UMAP:
        reducer = umap.UMAP(random_state=random_state)
        try:
            coords = reducer.fit_transform(X)
        except Exception:
            return X[:, :2], label_arr
    else:
        # ensure perplexity < n_samples
        px = min(30, max(2, n - 1, n // 3))
        px = min(px, max(2, n - 1))
        reducer = TSNE(n_components=2, random_state=random_state, init="pca", perplexity=px)
        try:
            coords = reducer.fit_transform(X)
        except Exception:
            return X[:, :2], label_arr
    return coords, label_arr


def plot_scatter(coords, labels, title, save_path):
    plt.figure(figsize=(6, 6))
    uniq = np.unique(labels)
    for u in uniq:
        idx = labels == u
        plt.scatter(coords[idx, 0], coords[idx, 1], s=10, label=str(u), alpha=0.7)
    plt.legend(markerscale=2, fontsize=8)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_heatmap(arr, title, save_path):
    plt.figure(figsize=(5, 4))
    try:
        # allow shape (1,H,W) by squeezing channel
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        elif arr.ndim == 3 and arr.shape[0] > 1:
            # 多通道时取均值，避免 imshow 维度报错
            arr = arr.mean(axis=0)
    except Exception:
        pass
    plt.imshow(arr, cmap="hot", aspect="auto")
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_graph(edge_weights, node_scores, out_path, title="GNN graph"):
    # edge_weights: list of (u, v, w)
    # node_scores: dict node->score
    G = nx.DiGraph()
    for u, v, w in edge_weights:
        G.add_edge(u, v, weight=w)
    for n, s in node_scores.items():
        G.add_node(n, score=s)
    pos = nx.spring_layout(G, seed=0)
    scores = np.array([node_scores.get(n, 0.0) for n in G.nodes()])
    node_colors = plt.cm.Blues(0.3 + 0.7 * (scores - scores.min()) / (scores.ptp() + 1e-6))
    widths = [max(0.5, 3.0 * w) for _, _, w in G.edges(data="weight")]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=400)
    nx.draw_networkx_labels(G, pos, font_size=8)
    nx.draw_networkx_edges(G, pos, width=widths, edge_color="gray", arrows=True, arrowstyle="-|>")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def compute_teacher_proto_scores(teacher, patches, proto_img_embs):
    if teacher is None or proto_img_embs is None:
        return None
    # 维度不匹配则直接跳过
    try:
        t_dim = teacher.img_proj.out_features
        if t_dim != proto_img_embs.shape[1]:
            warnings.warn(f"Skip teacher heatmap: teacher dim {t_dim} != proto dim {proto_img_embs.shape[1]}")
            return None
    except Exception:
        pass
    device = proto_img_embs.device
    teacher = teacher.to(device)
    teacher.eval()
    with torch.no_grad():
        v = teacher.img_encoder(patches)
        v = F.normalize(v, dim=1)
        scores = torch.matmul(v, proto_img_embs.t())  # (P, num_proto)
    return scores


def main():
    parser = argparse.ArgumentParser(description="Prototype/patch visualization for first N samples")
    parser.add_argument('--data_path', required=True, type=str)
    parser.add_argument('--ckpt', required=True, type=str)
    parser.add_argument('--prototype_lib_path', required=True, type=str)
    parser.add_argument('--knowledge_graph_path', default=None, type=str)
    parser.add_argument('--teacher_ckpt', default=None, type=str)
    parser.add_argument('--output_dir', default='./runs/vis_proto', type=str)
    parser.add_argument('--split', default='test', type=str)
    parser.add_argument('--num_samples', default=5, type=int)
    parser.add_argument('--sample_timesteps', default=None, type=int)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--patch_size', default=32, type=int)
    parser.add_argument('--stride', default=64, type=int)
    parser.add_argument('--topk', default=3, type=int)
    parser.add_argument('--reduce', default='tsne', choices=['tsne', 'umap'])
    args = parser.parse_args()

    device = torch.device(args.device)
    out_root = ensure_dir(args.output_dir)
    ensure_dir(os.path.join(out_root, 'tsne'))
    ensure_dir(os.path.join(out_root, 'heatmap'))
    ensure_dir(os.path.join(out_root, 'graph'))

    # dataset
    dataset = LRDataset(args.data_path, split=args.split)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    # model & matcher
    model, meta = build_model_from_ckpt(args.ckpt, device, sample_timesteps_override=args.sample_timesteps)
    # 为避免 teacher_ckpt 触发 matcher 的 teacher 前向(需要 mask)，这里给 matcher 构造不带 teacher 的参数副本
    args_for_modules = copy.deepcopy(args)
    if hasattr(args_for_modules, 'teacher_ckpt'):
        args_for_modules.teacher_ckpt = None
    matcher, kg_refiner, proto_centroids = build_condition_modules(args_for_modules, device)

    # load prototypes (for plotting)
    with open(args.prototype_lib_path, 'rb') as f:
        proto_lib = pickle.load(f)
    # flatten proto embeddings
    proto_emb_list = []
    proto_sid_list = []
    for sid, plist in proto_lib.items():
        for p in plist:
            if 'embedding' in p:
                proto_emb_list.append(np.asarray(p['embedding'], dtype=np.float32))
                proto_sid_list.append(int(sid))
    proto_emb_np = np.stack(proto_emb_list) if len(proto_emb_list) > 0 else None
    proto_emb = torch.from_numpy(proto_emb_np).float().to(device) if proto_emb_np is not None else None

    # teacher encoder (optional)
    teacher = load_teacher(args.teacher_ckpt, device, embed_dim=proto_emb.shape[1] if proto_emb is not None else None)

    all_proto = []
    all_student = []
    all_teacher = []
    all_labels_proto = []
    all_labels_student = []
    all_labels_teacher = []

    for idx, batch in enumerate(loader):
        if idx >= args.num_samples:
            break
        lr = batch['lr'].to(device)
        h, w = lr.shape[2:]
        # derive desired cond channels from model meta
        desired_cond_ch = max(1, int(meta['img_channels']) - 3 - int(lr.shape[1]))
        cond_map = torch.zeros((lr.size(0), desired_cond_ch, h, w), device=device)

        # matcher forward to get patch embeddings and scores
        if matcher is not None:
            try:
                res = matcher(lr)
            except Exception as e:
                warnings.warn(f"Matcher failed on sample {idx}: {e}")
                res = None
            patch_embs = None
            patch_coords = None
            entity_scores = None
            teacher_embs = None
            if isinstance(res, (tuple, list)):
                if len(res) >= 3:
                    patch_embs, patch_coords, entity_scores = res[0], res[1], res[2]
                if len(res) >= 4:
                    teacher_embs = res[3]
            # fallback cond_map using matcher+kg
            cond_map = build_cond_map_from_modules(lr, matcher, kg_refiner, desired_cond_ch, args.patch_size, args.stride, proto_centroids)
        else:
            patch_embs = None
            patch_coords = None
            entity_scores = None
            teacher_embs = None

        # collect embeddings for TSNE/UMAP
        if patch_embs is not None and patch_embs.numel() > 0:
            all_student.append(patch_embs.detach().cpu().numpy())
            all_labels_student.append(np.zeros((patch_embs.shape[0],), dtype=np.int32))
        if teacher_embs is not None:
            all_teacher.append(teacher_embs.detach().cpu().numpy())
            all_labels_teacher.append(np.zeros((teacher_embs.shape[0],), dtype=np.int32))
        if proto_emb_np is not None:
            all_proto.append(proto_emb_np)
            all_labels_proto.append(np.array(proto_sid_list))

        # similarity heatmaps (student and teacher vs prototypes)
        if patch_embs is not None and proto_emb is not None:
            student_sim = torch.matmul(F.normalize(patch_embs, dim=1), proto_emb.t())  # (P, N_proto)
            # scatter top-k per entity to map
            scores = student_sim.max(dim=1).values.unsqueeze(1)  # (P,1)
            cond_student = scatter_scores_to_map(scores, args.patch_size, args.stride, h, w, device)
            plot_heatmap(cond_student.cpu().numpy(), f"student heat {idx}", os.path.join(out_root, 'heatmap', f'student_{idx}.png'))
        if teacher is not None and proto_emb is not None and patch_embs is not None:
            # build patch tensors for teacher scoring: need original patches from matcher; if matcher not returned, skip
            # Here we approximate by re-extracting patches from lr (first image only)
            # note: this is a best-effort visualization, not affecting model outputs.
            try:
                from train_stage1.LREmbeddingMatcher import LREmbeddingMatcher
                # Reuse matcher extract_patches to get patch tensors
                patches_np = matcher.extract_patches(lr[0].detach().cpu())
                patch_tensors = []
                coords = []
                for patch, (x, y) in patches_np:
                    t = matcher.transform(patch).to(device)
                    patch_tensors.append(t)
                    coords.append((x, y))
                if len(patch_tensors) > 0:
                    patch_batch = torch.stack(patch_tensors, dim=0)
                    teacher_scores = compute_teacher_proto_scores(teacher, patch_batch, proto_emb)
                    if teacher_scores is not None:
                        scores = teacher_scores.max(dim=1).values.unsqueeze(1)
                        cond_teacher = scatter_scores_to_map(scores, args.patch_size, args.stride, h, w, device)
                        plot_heatmap(cond_teacher.cpu().numpy(), f"teacher heat {idx}", os.path.join(out_root, 'heatmap', f'teacher_{idx}.png'))
            except Exception as e:
                warnings.warn(f"Teacher heatmap failed: {e}")

        # GNN graph + entity/cond map前后对比
        if kg_refiner is not None and entity_scores is not None and entity_scores.numel() > 0:
            try:
                es = entity_scores[0].to(device)  # L x E
                es_np = es.detach().cpu().numpy()
                init_scores = es_np.mean(axis=0)
                # cond_map before GNN
                cond_before = scatter_scores_to_map(es.detach(), args.patch_size, args.stride, h, w, device)

                with torch.no_grad():
                    refined_entity, refined_patch = kg_refiner(es, patch_embs=patch_embs[0].detach() if patch_embs is not None else None)
                refined_np = refined_entity.detach().cpu().numpy()
                # cond_map after GNN (用 patch 修正分数)
                cond_after = scatter_scores_to_map(refined_patch.detach(), args.patch_size, args.stride, h, w, device)

                plot_heatmap(cond_before.cpu().numpy(), f"KG before {idx}", os.path.join(out_root, 'heatmap', f'kg_before_{idx}.png'))
                plot_heatmap(cond_after.cpu().numpy(), f"KG after {idx}", os.path.join(out_root, 'heatmap', f'kg_after_{idx}.png'))

                # bar plot for entity scores
                plt.figure(figsize=(6, 3))
                x = np.arange(init_scores.shape[0])
                plt.bar(x - 0.15, init_scores, width=0.3, label='before')
                plt.bar(x + 0.15, refined_np, width=0.3, label='after')
                plt.title(f'Entity scores {idx}')
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(out_root, 'graph', f'entity_bar_{idx}.png'), dpi=200)
                plt.close()

                # graph view
                with open(args.knowledge_graph_path, 'r') as f:
                    kg_json = json.load(f)
                edges = []
                for rel in kg_json.get('relations', []):
                    try:
                        u = int(rel['from']); v = int(rel['to']);
                        edges.append((u, v, 1.0))
                    except Exception:
                        continue
                if not edges:
                    E = refined_np.shape[0]
                    edges = [(i, j, 1.0) for i in range(E) for j in range(E) if i != j]
                node_scores_init = {i: float(init_scores[i]) for i in range(init_scores.shape[0])}
                node_scores_ref = {i: float(refined_np[i]) for i in range(refined_np.shape[0])}
                plot_graph(edges, node_scores_init, os.path.join(out_root, 'graph', f'init_{idx}.png'), title=f'KG init {idx}')
                plot_graph(edges, node_scores_ref, os.path.join(out_root, 'graph', f'refined_{idx}.png'), title=f'KG refined {idx}')
            except Exception as e:
                warnings.warn(f"Graph viz failed: {e}")

    # TSNE / UMAP global plots
    if all_proto:
        coords, labels = reduce_embeddings(all_proto, all_labels_proto, method=args.reduce)
        plot_scatter(coords, labels, f'Prototypes ({args.reduce})', os.path.join(out_root, 'tsne', f'prototypes_{args.reduce}.png'))
    if all_student:
        coords, labels = reduce_embeddings(all_student, all_labels_student, method=args.reduce)
        plot_scatter(coords, labels, f'Student patches ({args.reduce})', os.path.join(out_root, 'tsne', f'student_{args.reduce}.png'))
    if all_teacher:
        coords, labels = reduce_embeddings(all_teacher, all_labels_teacher, method=args.reduce)
        plot_scatter(coords, labels, f'Teacher patches ({args.reduce})', os.path.join(out_root, 'tsne', f'teacher_{args.reduce}.png'))

    print(f"Visualization saved to {out_root}")


if __name__ == '__main__':
    main()

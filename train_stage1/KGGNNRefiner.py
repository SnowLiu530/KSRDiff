import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# try to import PyG TransformerConv; if unavailable, we'll fallback to an MLP-based implementation
try:
    from torch_geometric.nn import TransformerConv
    _HAS_PYG = True
except Exception:
    TransformerConv = None
    _HAS_PYG = False


class KGGNNRefiner(nn.Module):
    """
    基于知识图谱的 Transformer-style GNN，用实体作为节点。
    输入：patch_scores (P x E)，patch_embs (P x D)
    输出：entity-level 修正分数 (E,)，以及可选的 patch-level 修正 (P x E)
    """
    def __init__(self, knowledge_graph_path, entity_ids=None, in_dim=128, hidden_dim=128, num_layers=2, device='cuda'):
        super().__init__()
        self.device = device
        # 加载知识图谱
        with open(knowledge_graph_path, 'r') as f:
            self.kg = json.load(f)
        # entity_ids: 若提供则使用，否则从 kg 读取并顺序化
        if entity_ids is not None:
            self.entity_ids = list(entity_ids)
        else:
            self.entity_ids = list(self.kg.get('entities', {}).keys())
        self.num_entities = len(self.entity_ids)

        # 输入投影：把聚合后的 node feature 投影到 hidden_dim
        self.in_proj = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()

        # 构建 GNN（如果有 torch_geometric），否则构建等价的 MLP 回退实现
        self.convs = nn.ModuleList()
        if _HAS_PYG:
            for i in range(num_layers):
                in_c = hidden_dim
                self.convs.append(TransformerConv(in_c, hidden_dim, heads=4, concat=False))
        else:
            # MLP fallback: 多层线性 + ReLU
            for i in range(num_layers):
                self.convs.append(nn.Linear(hidden_dim, hidden_dim))
        self.fc_out = nn.Linear(hidden_dim, 1)

        self.to(device)

    def build_graph(self):
        edge_index = []
        for rel in self.kg.get('relations', []):
            try:
                src = int(rel['from'])
                tgt = int(rel['to'])
            except Exception:
                continue
            edge_index.append([src, tgt])
            edge_index.append([tgt, src])
        if len(edge_index) == 0:
            # fallback: fully connected
            idxs = list(range(self.num_entities))
            for i in idxs:
                for j in idxs:
                    if i != j:
                        edge_index.append([i, j])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(self.device)
        return edge_index

    def build_adjacency(self):
        """Return dense adjacency matrix (E x E) normalized row-wise."""
        edge_index = []
        for rel in self.kg.get('relations', []):
            try:
                src = int(rel['from'])
                tgt = int(rel['to'])
            except Exception:
                continue
            edge_index.append([src, tgt])
            edge_index.append([tgt, src])
        if len(edge_index) == 0:
            idxs = list(range(self.num_entities))
            for i in idxs:
                for j in idxs:
                    if i != j:
                        edge_index.append([i, j])
        E = self.num_entities
        A = torch.zeros((E, E), device=self.device)
        for src, tgt in edge_index:
            if 0 <= src < E and 0 <= tgt < E:
                A[src, tgt] = 1.0
        # row-normalize
        row_sum = A.sum(dim=1, keepdim=True)
        row_sum[row_sum == 0] = 1.0
        A = A / row_sum
        return A

    def apply_kg_constraints(self, patch_scores_np, coords_np):
        # 保留原有约束逻辑（可选性地作为后处理）
        scores = patch_scores_np.copy()
        entity_map = self.entity_ids
        # defensive checks
        try:
            body_idx = entity_map.index('0')
            tail_idx = entity_map.index('1')
            body_x = coords_np[:,0][body_idx] if body_idx < coords_np.shape[0] else None
            tail_x = coords_np[:,0][tail_idx] if tail_idx < coords_np.shape[0] else None
            if body_x is not None and tail_x is not None:
                if tail_x >= body_x:
                    scores[:, tail_idx] *= 0.1
        except Exception:
            pass
        return scores

    def forward(self, patch_scores, patch_embs=None, patch_coords=None, proto_centroids=None, sigma=50.0):
        """
        patch_scores: P x E
        patch_embs: P x D (optional, recommended)
        proto_centroids: list of (x,y) per entity (optional)
        返回: entity_scores E (tensor)，以及 patch_corrected (P x E)
        """
        patch_scores = patch_scores.to(self.device)
        P, E = patch_scores.shape
        assert E == self.num_entities, f'Entity count mismatch: {E} vs {self.num_entities}'

        # 聚合 patch -> entity node features
        if patch_embs is None:
            # use patch_scores as features if emb not provided
            # transpose -> E x P
            node_feats = patch_scores.t()
            node_feats = node_feats.unsqueeze(-1)  # E x P x 1
            node_feats = node_feats.mean(dim=1)  # E x 1
        else:
            # spatial weighting by proto_centroids if provided
            w = patch_scores  # P x E
            if (proto_centroids is not None) and (patch_coords is not None):
                coords = torch.tensor(patch_coords, device=patch_embs.device).float()  # P x 2
                centroids = torch.tensor([c if c is not None else (0.0,0.0) for c in proto_centroids], device=patch_embs.device).float()  # E x 2
                dists = torch.cdist(coords, centroids)  # P x E
                spatial_w = torch.exp(-(dists**2) / (2 * (sigma**2)))
                w = w * spatial_w
            # normalize per-entity
            denom = w.sum(dim=0, keepdim=True) + 1e-8
            w_norm = w / denom
            # node_feats: E x D = (w_norm.T @ patch_embs)
            node_feats = torch.matmul(w_norm.t(), patch_embs)

        # project to hidden dim
        node_feats = self.in_proj(node_feats)

        edge_index = self.build_graph()
        x = node_feats
        # apply GNN or MLP-with-adjacency fallback
        if _HAS_PYG:
            for conv in self.convs:
                x = conv(x, edge_index)
                x = F.relu(x)
        else:
            # adjacency-based message passing: x <- A @ x ; then linear + relu
            A = self.build_adjacency()
            for lin in self.convs:
                # message aggregation
                x = torch.matmul(A, x)
                x = lin(x)
                x = F.relu(x)
        entity_out = torch.sigmoid(self.fc_out(x)).squeeze(1)  # E

        # map back to patch-level if needed
        patch_corrected = patch_scores * entity_out.unsqueeze(0)
        return entity_out, patch_corrected

# ------------------ 主函数示例 ------------------
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 假设我们有 patch 对每个实体的初始置信度
    patch_num = 5
    entity_num = 4
    patch_scores = torch.rand(patch_num, entity_num)  # 5 patch x 4 实体

    # 假设每个 patch 对应的中心坐标 (x,y)
    coords = np.random.randn(patch_num, 2)

    # 初始化 GNN refiner
    kg_path = 'knowledge_graph.json'
    gnn_refiner = KGGNNRefiner(kg_path, in_dim=1, hidden_dim=64, num_layers=2, device=device)

    # 修正初始置信度
    patch_scores_corrected = gnn_refiner.forward(patch_scores)

    # 应用 KG 显式约束
    patch_scores_corrected_np = patch_scores_corrected.detach().cpu().numpy()
    patch_scores_final = gnn_refiner.apply_kg_constraints(patch_scores_corrected_np, coords)

    print("修正后的 patch_scores：\n", patch_scores_final)

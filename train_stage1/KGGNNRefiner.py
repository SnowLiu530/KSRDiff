import json
import torch
import torch.nn as nn
import torch.nn.functional as F


class KGGNNRefiner(nn.Module):
    """
    Aircraft Knowledge Graph-GNN.

    Inputs
    ------
    patch_probs : Tensor [P, E]
        ESSD输出的patch-to-entity soft assignment P^S_{i,m}.

    patch_embs : Tensor [P, D]
        Transformer contextualized patch representations y_i.

    Outputs
    -------
    node_feats : Tensor [E, D_h]
        Knowledge-enhanced entity representations h_m^(L_g).

    patch_condition : Tensor [P, D_h]
        Knowledge-enhanced patch representations projected back from entity
        nodes, which can subsequently be reshaped/projected to the diffusion
        conditioning map C.
    """

    def __init__(
        self,
        knowledge_graph_path,
        entity_ids=None,
        in_dim=128,
        hidden_dim=128,
        num_layers=2,
        bidirectional=True,
        device="cuda",
    ):
        super().__init__()

        self.device = torch.device(device)
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # ---------------------------------------------------------
        # 1. Load Aircraft KG
        # ---------------------------------------------------------
        with open(knowledge_graph_path, "r") as f:
            self.kg = json.load(f)

        if entity_ids is not None:
            self.entity_ids = [str(x) for x in entity_ids]
        else:
            entities = self.kg.get("entities", {})

            if isinstance(entities, dict):
                self.entity_ids = [str(k) for k in entities.keys()]
            elif isinstance(entities, list):
                self.entity_ids = [
                    str(e["id"]) if isinstance(e, dict) else str(e)
                    for e in entities
                ]
            else:
                raise ValueError("Invalid `entities` field in knowledge graph.")

        self.num_entities = len(self.entity_ids)

        if self.num_entities == 0:
            raise ValueError("Aircraft KG contains no entities.")

        self.entity_to_idx = {
            entity_id: idx
            for idx, entity_id in enumerate(self.entity_ids)
        }

        # ---------------------------------------------------------
        # 2. Parse typed KG relations
        # ---------------------------------------------------------
        self.edges = []
        relation_types = set()

        for rel in self.kg.get("relations", []):
            src_id = str(rel["from"])
            dst_id = str(rel["to"])

            # Support common JSON field names
            rel_type = (
                rel.get("type")
                or rel.get("relation")
                or rel.get("name")
            )

            if rel_type is None:
                raise ValueError(
                    f"Relation {rel} has no relation type."
                )

            rel_type = str(rel_type)

            if src_id not in self.entity_to_idx:
                raise ValueError(f"Unknown entity id: {src_id}")
            if dst_id not in self.entity_to_idx:
                raise ValueError(f"Unknown entity id: {dst_id}")

            src = self.entity_to_idx[src_id]
            dst = self.entity_to_idx[dst_id]

            self.edges.append((src, dst, rel_type))
            relation_types.add(rel_type)

            # If the KG relations are treated as symmetric connectivity,
            # use the same relation transform in the reverse direction.
            if bidirectional:
                self.edges.append((dst, src, rel_type))

        if len(self.edges) == 0:
            raise ValueError(
                "Aircraft KG contains no valid relations. "
                "KG-GNN must not fall back to a fully connected graph."
            )

        self.relation_types = sorted(relation_types)

        # ---------------------------------------------------------
        # 3. Input projection
        # ---------------------------------------------------------
        if in_dim != hidden_dim:
            self.in_proj = nn.Linear(in_dim, hidden_dim)
        else:
            self.in_proj = nn.Identity()

        # ---------------------------------------------------------
        # 4. Relation-specific transformations W_r^(l)
        #
        # Each graph layer has its own W_r for every relation type.
        # ---------------------------------------------------------
        self.rel_transforms = nn.ModuleList()

        for _ in range(num_layers):
            layer_transforms = nn.ModuleDict({
                r: nn.Linear(
                    hidden_dim,
                    hidden_dim,
                    bias=False
                )
                for r in self.relation_types
            })
            self.rel_transforms.append(layer_transforms)

        # Optional normalization after residual node update.
        # If this is not present in your actual implementation,
        # replace with nn.Identity().
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])

        self.to(self.device)

    # -------------------------------------------------------------
    # ESSD patch -> entity aggregation
    #
    # h_m^(0) = sum_i P^S_{i,m} y_i
    #
    # Here normalized=True uses a weighted mean for numerical
    # stability. Set False if you want literal Eq. (23).
    # -------------------------------------------------------------
    def aggregate_entities(
        self,
        patch_probs,
        patch_embs,
        normalized=True,
        eps=1e-8,
    ):
        """
        patch_probs : [P, E]
        patch_embs  : [P, D]
        """

        if patch_probs.dim() != 2:
            raise ValueError("patch_probs must have shape [P, E].")

        if patch_embs.dim() != 2:
            raise ValueError("patch_embs must have shape [P, D].")

        P, E = patch_probs.shape

        if E != self.num_entities:
            raise ValueError(
                f"Entity number mismatch: input={E}, KG={self.num_entities}"
            )

        if patch_embs.shape[0] != P:
            raise ValueError(
                "patch_probs and patch_embs must contain the same number "
                "of patches."
            )

        patch_probs = patch_probs.to(self.device)
        patch_embs = patch_embs.to(self.device)

        if normalized:
            denom = patch_probs.sum(dim=0, keepdim=True) + eps
            weights = patch_probs / denom
        else:
            weights = patch_probs

        # [E, P] @ [P, D] -> [E, D]
        h0 = weights.transpose(0, 1) @ patch_embs

        return self.in_proj(h0)

    # -------------------------------------------------------------
    # One relation-aware KG layer
    #
    # beta_mj^(l)
    #   = sigmoid(h_m^T W_r h_j)
    #
    # m_m^(l)
    #   = sum_{j:A_mj=1}
    #       beta_mj W_r h_j
    #
    # h_m^(l+1)
    #   = GELU(h_m^(l) + m_m^(l))
    # -------------------------------------------------------------
    def kg_layer(self, x, layer_idx):
        """
        x : [E, D_h]
        """

        E, D = x.shape
        messages = torch.zeros_like(x)

        layer_transforms = self.rel_transforms[layer_idx]

        # Optional: store beta for visualization / analysis
        edge_betas = []

        for src, dst, rel_type in self.edges:
            # Convention:
            # src = j (source/neighbour)
            # dst = m (target node)

            h_j = x[src]   # [D]
            h_m = x[dst]   # [D]

            # W_r h_j
            transformed_j = layer_transforms[rel_type](h_j)

            # beta_mj = sigmoid(h_m^T W_r h_j)
            beta = torch.sigmoid(
                torch.dot(h_m, transformed_j)
            )

            # m_m += beta_mj W_r h_j
            messages[dst] = (
                messages[dst]
                + beta * transformed_j
            )

            edge_betas.append({
                "src": src,
                "dst": dst,
                "relation": rel_type,
                "beta": beta,
            })

        # Residual node update
        x_new = F.gelu(x + messages)

        # Can be removed if not used in the actual implementation
        x_new = self.norms[layer_idx](x_new)

        return x_new, edge_betas

    # -------------------------------------------------------------
    # Entity -> patch projection
    #
    # Conceptually:
    # C_i = sum_m P^S_{i,m} h_m^(L_g)
    #
    # The resulting patch representation can then be reshaped by Phi
    # into the spatial condition map used by the diffusion U-Net.
    # -------------------------------------------------------------
    @staticmethod
    def project_to_patches(patch_probs, node_feats):
        """
        patch_probs : [P, E]
        node_feats  : [E, D_h]

        return      : [P, D_h]
        """

        return patch_probs @ node_feats

    # -------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------
    def forward(
        self,
        patch_probs,
        patch_embs,
        return_attention=False,
    ):
        """
        patch_probs : [P, E]
            ESSD entity-assignment distribution P^S.

        patch_embs : [P, D]
            Contextualized patch embeddings y_i.
        """

        patch_probs = patch_probs.to(self.device)
        patch_embs = patch_embs.to(self.device)

        # Eq. (23): patch -> entity aggregation
        x = self.aggregate_entities(
            patch_probs,
            patch_embs,
            normalized=True,
        )

        all_edge_betas = []

        # Eq. (24)-(27): typed relation-aware graph reasoning
        for l in range(self.num_layers):
            x, edge_betas = self.kg_layer(x, l)
            all_edge_betas.append(edge_betas)

        # Knowledge-enhanced entity nodes:
        # h_m^(L_g)
        node_feats = x

        # Project graph knowledge back to patch domain
        patch_condition = self.project_to_patches(
            patch_probs,
            node_feats,
        )

        if return_attention:
            return node_feats, patch_condition, all_edge_betas

        return node_feats, patch_condition


# ================================================================
# Example
# ================================================================
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    P = 64
    E = 4
    D = 128

    # ESSD output:
    # each patch has a soft probability over
    # fuselage / left wing / right wing / tail
    patch_logits = torch.randn(P, E, device=device)
    patch_probs = torch.softmax(patch_logits, dim=-1)

    # Transformer-contextualized patch embeddings y_i
    patch_embs = torch.randn(P, D, device=device)

    kg_path = "knowledge_graph.json"

    kg_gnn = KGGNNRefiner(
        knowledge_graph_path=kg_path,
        entity_ids=["0", "1", "2", "3"],
        in_dim=D,
        hidden_dim=128,
        num_layers=2,
        bidirectional=True,
        device=device,
    )

    node_feats, patch_condition, edge_betas = kg_gnn(
        patch_probs,
        patch_embs,
        return_attention=True,
    )

    print("Knowledge-enhanced entity nodes:")
    print(node_feats.shape)
    # [4, 128]

    print("Knowledge-enhanced patch representation:")
    print(patch_condition.shape)
    # [64, 128]

    for layer_id, layer_edges in enumerate(edge_betas):
        print(f"\nLayer {layer_id}")

        for edge in layer_edges:
            print(
                f"{edge['src']} -> {edge['dst']} "
                f"[{edge['relation']}], "
                f"beta={edge['beta'].item():.4f}"
            )

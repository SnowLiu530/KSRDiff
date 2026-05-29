import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TeacherImageWrapper(nn.Module):
    """
    Wrapper that exposes a teacher model in a simple image/mask embedding API.
    The wrapped teacher is frozen (eval, requires_grad=False).
    When called with only img_patches returns image embeddings v; when called with (img,mask)
    returns (v,g) as teacher provides.
    """
    def __init__(self, teacher_model, device='cuda'):
        super().__init__()
        self.teacher = teacher_model.to(device)
        self.device = device
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, img_patches, mask_patches=None):
        # Accepts (N, C, H, W) or (B, L, C, H, W) flattened externally
        if mask_patches is None:
            out = self.teacher(img_patches)
        else:
            out = self.teacher(img_patches, mask_patches)
        return out


def load_prototypes_from_file(path, device='cuda'):
    """Load prototype lib (pickle) and return tensors + stable sid order.

    Returns:
      proto_img: (M, D) stacked all prototypes across sids
      proto_mask: (M, D) stacked masks (or reused img emb)
      sid_list: list of sids (entity order)
      proto_centroids: list of centroid tuples per sid (or None)

    Note: This function flattens all prototypes into two big tensors. It's caller's
    responsibility to also keep per-sid counts if needed.
    """
    with open(path, 'rb') as f:
        proto = pickle.load(f)

    # stable sid order
    sid_list = sorted(list(proto.keys()), key=lambda x: str(x))

    img_list = []
    mask_list = []
    # per-sid centroid (compute mean of prototype centroids if available)
    proto_centroids = []
    for sid in sid_list:
        plist = proto.get(sid, [])
        centroids = []
        for p in plist:
            emb = p.get('embedding', None)
            if emb is None:
                continue
            img_list.append(np.asarray(emb, dtype=np.float32))
            # prefer explicit mask embedding; do NOT use polygon/centroid as mask vector
            mask_emb = p.get('mask_embedding', None)
            if mask_emb is None:
                # fallback: reuse image embedding
                mask_list.append(np.asarray(emb, dtype=np.float32))
            else:
                mask_list.append(np.asarray(mask_emb, dtype=np.float32))

            # collect centroid if present (expect [x,y] or similar)
            c = p.get('centroid', None)
            if c is not None:
                try:
                    centroids.append((float(c[0]), float(c[1])))
                except Exception:
                    pass

        if len(centroids) > 0:
            # average centroid for this sid
            xs = [c[0] for c in centroids]
            ys = [c[1] for c in centroids]
            proto_centroids.append((float(np.mean(xs)), float(np.mean(ys))))
        else:
            proto_centroids.append(None)

    if len(img_list) == 0:
        raise RuntimeError(f'No valid prototypes found in {path}')

    proto_img = torch.from_numpy(np.stack(img_list, axis=0)).float().to(device)
    proto_mask = torch.from_numpy(np.stack(mask_list, axis=0)).float().to(device)
    proto_img = F.normalize(proto_img, dim=1)
    proto_mask = F.normalize(proto_mask, dim=1)
    return proto_img, proto_mask, sid_list, proto_centroids


def student_mask_from_prototypes(s_v, proto_img_embs, proto_mask_embs, temp=0.1):
    """
    s_v: (B, L, D) or (B, D) or (N, D). Returns s_g of shape same-prefix with last dim D.
    proto_img_embs: (M, D), proto_mask_embs: (M, D)
    Returns: (s_g, weights)
    """
    orig_shape = s_v.shape
    if s_v.dim() == 2:
        flat = s_v
        is_3d = False
    elif s_v.dim() == 3:
        B, L, D = s_v.shape
        flat = s_v.view(B * L, D)
        is_3d = True
    else:
        raise ValueError('s_v must be 2D or 3D')

    # logits: (N, M)
    logits = torch.matmul(flat, proto_img_embs.t())
    weights = F.softmax(logits / temp, dim=1)
    s_g_flat = torch.matmul(weights, proto_mask_embs)

    if is_3d:
        s_g = s_g_flat.view(B, L, -1)
    else:
        s_g = s_g_flat
    return s_g, weights


class PrototypeLoader:
    """Convenience wrapper for loading prototypes and exposing tensors."""
    def __init__(self, path=None, device='cuda'):
        self.path = path
        self.device = device
        self.proto_img = None
        self.proto_mask = None

    def load(self, path=None):
        p = path or self.path
        if p is None:
            raise ValueError('No path provided to PrototypeLoader.load')
        self.proto_img, self.proto_mask = load_prototypes_from_file(p, device=self.device)
        return self.proto_img, self.proto_mask


__all__ = [
    'TeacherImageWrapper',
    'load_prototypes_from_file',
    'student_mask_from_prototypes',
    'PrototypeLoader',
]
import pickle
import numpy as np
import torch
import torch.nn.functional as F


class PrototypeLoader:
    @staticmethod
    def load(proto_path, device=None):
        """Load prototype dict from pickle and return (proto_img, proto_mask) tensors.

        proto_path: path to prototype_lib.pkl
        device: torch device or None
        Returns: proto_img (N_proto x D), proto_mask (N_proto x D)
        """
        if device is None:
            device = torch.device('cpu')
        with open(proto_path, 'rb') as f:
            proto = pickle.load(f)

        img_list = []
        mask_list = []
        for sid, plist in proto.items():
            for p in plist:
                emb = p.get('embedding', None)
                if emb is None:
                    continue
                img_list.append(np.asarray(emb, dtype=np.float32))
                mask_emb = p.get('mask_embedding', None)
                if mask_emb is not None:
                    me = np.asarray(mask_emb)
                    if me.ndim == 1 and me.shape[0] == len(emb):
                        mask_list.append(me.astype(np.float32))
                    else:
                        mask_list.append(np.asarray(emb, dtype=np.float32))
                else:
                    mask_list.append(np.asarray(emb, dtype=np.float32))

        if len(img_list) == 0:
            raise RuntimeError('No prototypes found in %s' % proto_path)

        proto_img = torch.from_numpy(np.stack(img_list, axis=0)).float().to(device)
        proto_mask = torch.from_numpy(np.stack(mask_list, axis=0)).float().to(device)
        proto_img = F.normalize(proto_img, dim=1)
        proto_mask = F.normalize(proto_mask, dim=1)
        return proto_img, proto_mask


class TeacherImageWrapper:
    """Wrap a teacher model to provide normalized image/mask embeddings and freeze params.

    Usage:
        wrapper = TeacherImageWrapper(teacher_model)
        wrapper.eval()
        v,g = wrapper(patches, masks)
    """
    def __init__(self, teacher_model):
        self.model = teacher_model
        # freeze params
        for p in getattr(self.model, 'parameters', lambda: [])():
            try:
                p.requires_grad = False
            except Exception:
                pass

    def eval(self):
        try:
            self.model.eval()
        except Exception:
            pass
        return self

    def __call__(self, img_patches, mask_patches=None):
        """Call the wrapped teacher. Returns normalized (v, g) or v if teacher returns single.

        img_patches: (N, C, ps, ps)
        mask_patches: optional (N, 1, ps, ps)
        """
        with torch.no_grad():
            if mask_patches is None:
                out = self.model(img_patches)
            else:
                out = self.model(img_patches, mask_patches)

        if isinstance(out, (tuple, list)):
            v = out[0]
            g = out[1]
            v = F.normalize(v, dim=-1)
            g = F.normalize(g, dim=-1)
            return v, g
        else:
            v = out
            v = F.normalize(v, dim=-1)
            return v


class StudentMaskComputer:
    @staticmethod
    def compute(student_embs, proto_img, proto_mask, temp=0.1):
        """Compute student mask embeddings by soft-attention over prototypes.

        student_embs: B x L x D or L x D
        proto_img: N_proto x D
        proto_mask: N_proto x D
        returns: s_mask (B x L x D), weights (B*L x N_proto)
        """
        if student_embs.dim() == 2:
            student_embs = student_embs.unsqueeze(0)
        student_embs_norm = F.normalize(student_embs, dim=2)
        B, L, D = student_embs_norm.shape
        flat = student_embs_norm.view(B * L, D)
        logits = torch.matmul(flat, proto_img.t())
        weights = F.softmax(logits / float(temp), dim=1)
        s_mask_flat = torch.matmul(weights, proto_mask)
        s_mask = s_mask_flat.view(B, L, D)
        return s_mask, weights

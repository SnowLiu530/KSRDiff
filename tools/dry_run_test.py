import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import cv2

# paths (adjust if needed)
proto_path = '/mnt/share/liuxn/swinir_kg/prototype_lib/prototype_lib.pkl'
teacher_ckpt = '/mnt/share/liuxn/swinir_kg/train_stage1/teacher.pth'
data_lr_dir = '/mnt/share/liuxn/swinir_kg/dataset/train/LR'
data_img_dir = '/mnt/share/liuxn/swinir_kg/dataset/train/images'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device', device)

# find first pair
lr_files = sorted([os.path.join(data_lr_dir, f) for f in os.listdir(data_lr_dir) if os.path.isfile(os.path.join(data_lr_dir, f))])
img_files = sorted([os.path.join(data_img_dir, f) for f in os.listdir(data_img_dir) if os.path.isfile(os.path.join(data_img_dir, f))])
if len(lr_files) == 0 or len(img_files) == 0:
    raise RuntimeError('No LR or HR files found in dataset paths')

lr_path = lr_files[0]
img_path = img_files[0]
print('using', lr_path, img_path)

# read images
lr = cv2.imread(lr_path)
if lr is None:
    raise RuntimeError('failed to read '+lr_path)
lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0

# to tensor BxCxHxW, normalize to [-0.5,0.5] like train fallback
lr_t = torch.from_numpy(lr.transpose(2,0,1)).unsqueeze(0).float().to(device) - 0.5
img_t = torch.from_numpy(img.transpose(2,0,1)).unsqueeze(0).float().to(device) - 0.5

print('lr_t', lr_t.shape, 'img_t', img_t.shape)

# load prototype tools and prototypes (PrototypeLoader, TeacherImageWrapper, StudentMaskComputer)
import importlib.util
tools_path = os.path.join(os.getcwd(), 'ID-Blau', 'tools', 'prototype_tools.py')
spec_tools = importlib.util.spec_from_file_location('prototype_tools', tools_path)
proto_mod = importlib.util.module_from_spec(spec_tools)
spec_tools.loader.exec_module(proto_mod)
PrototypeLoader = proto_mod.PrototypeLoader
TeacherImageWrapper = proto_mod.TeacherImageWrapper
StudentMaskComputer = proto_mod.StudentMaskComputer

# load prototypes and infer embed dim
proto_img, proto_mask = PrototypeLoader.load(proto_path, device=device)
embed_dim = proto_img.shape[1]
print('embed_dim from prototype', embed_dim)

# import image encoder and matcher
sys.path.insert(0, os.getcwd())
from train_stage1.clip_part_align import ImageEncoder as ExternalImageEncoder
from train_stage1.LREmbeddingMatcher import LREmbeddingMatcher

img_encoder = ExternalImageEncoder(embed_dim).to(device)
# use patch_size 64 stride 64 for this test
matcher = LREmbeddingMatcher(img_encoder, proto_path, device=device, patch_size=64, stride=64)
matcher.eval()

with torch.no_grad():
    # LREmbeddingMatcher.forward expects a numpy HxWxC image (not a torch tensor)
    res = matcher(lr)
    if isinstance(res, (tuple, list)):
        patch_embs, patch_coords = res[0], res[1]
    else:
        patch_embs, patch_coords = res, None

print('patch_embs', patch_embs.shape)

# load teacher model
# load TeacherModel from ID-Blau/train_stage1/TeacherClass.py (avoid import side-effects)
import importlib.util
teacher_module_path = os.path.join(os.getcwd(), 'ID-Blau', 'train_stage1', 'TeacherClass.py')
spec = importlib.util.spec_from_file_location('teacher_module', teacher_module_path)
teacher_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(teacher_mod)
ExternalTeacherModel = getattr(teacher_mod, 'TeacherModel')
# TeacherModel signature: TeacherModel(img_dim=3, mask_dim=1, embed_dim=..)
teacher = ExternalTeacherModel(embed_dim=embed_dim).to(device)
# load checkpoint
sd = torch.load(teacher_ckpt, map_location='cpu')
if isinstance(sd, dict) and 'state_dict' in sd:
    sd = sd['state_dict']

try:
    teacher.load_state_dict(sd, strict=False)
    print('teacher ckpt loaded')
except Exception as e:
    print('warning loading teacher ckpt:', e)

# wrap teacher with the reusable wrapper (freezes params, returns normalized v/g)
teacher_wrapper = TeacherImageWrapper(teacher)
teacher_wrapper.eval()

# create patches from HR and run teacher on them
B, C, H, W = img_t.shape
ps = 64
stride = 64
unfold = torch.nn.Unfold(kernel_size=ps, stride=stride)
patches = unfold(img_t)  # B x (C*ps*ps) x L
patches = patches.transpose(1,2).contiguous()  # B x L x flat
B, L, flat = patches.shape
patches = patches.view(B*L, C, ps, ps).to(device)
print('num patches', B, L, 'patches tensor', patches.shape)

# run teacher in chunks via wrapper (returns normalized vectors)
out_list = []
out_mask_list = []
chunk = 256
with torch.no_grad():
    for i in range(0, patches.shape[0], chunk):
        ch = patches[i:i+chunk]
        zero_mask = torch.zeros((ch.size(0), 1, ch.size(2), ch.size(3)), device=ch.device)
        out = teacher_wrapper(ch, zero_mask)
        if isinstance(out, (tuple, list)):
            v, g = out
            out_list.append(v.detach().cpu())
            out_mask_list.append(g.detach().cpu())
        else:
            out_list.append(out.detach().cpu())

emb_all = torch.cat(out_list, dim=0)
emb_all = emb_all.view(B, L, emb_all.shape[-1]).to(device)
print('teacher image emb', emb_all.shape)
if len(out_mask_list) > 0:
    mask_all = torch.cat(out_mask_list, dim=0)
    mask_all = mask_all.view(B, L, mask_all.shape[-1]).to(device)
    print('teacher mask emb', mask_all.shape)
else:
    mask_all = None
    print('teacher returned no mask emb')

print('proto_img', proto_img.shape, 'proto_mask', proto_mask.shape)

# compute student mask from prototypes using StudentMaskComputer
student_embs = patch_embs
s_mask, weights = StudentMaskComputer.compute(student_embs, proto_img, proto_mask, temp=0.1)
print('s_mask', s_mask.shape, 'weights', weights.shape)

# compute quick cosine distill losses
s_norm = F.normalize(student_embs, dim=2)
Ds = s_norm.shape[2]
sv = s_norm.view(-1, Ds)
tv = emb_all.view(-1, Ds)
loss_img = 1.0 - (sv * tv).sum(dim=1).mean()
print('img distill loss', float(loss_img))
if mask_all is not None:
    sm = s_mask.view(-1, Ds)
tm = mask_all.view(-1, Ds) if mask_all is not None else None
if tm is not None:
    loss_mask = 1.0 - (sm * tm).sum(dim=1).mean()
    print('mask distill loss', float(loss_mask))

print('dry-run complete')

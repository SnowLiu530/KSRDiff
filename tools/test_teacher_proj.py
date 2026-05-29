import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import importlib.util
import os

# load prototype_tools from ID-Blau/tools
tools_path = os.path.join(os.getcwd(), 'ID-Blau', 'tools', 'prototype_tools.py')
spec_tools = importlib.util.spec_from_file_location('prototype_tools', tools_path)
proto_mod = importlib.util.module_from_spec(spec_tools)
spec_tools.loader.exec_module(proto_mod)
StudentMaskComputer = proto_mod.StudentMaskComputer


def simulate_projection_creation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, L = 1, 2
    D_s = 128
    D_t = 256
    # fake student and teacher embeddings
    student_embs = torch.randn(B, L, D_s, device=device)
    teacher_embs = torch.randn(B, L, D_t, device=device)

    # prepare optimizer with a dummy param
    dummy_param = nn.Parameter(torch.randn(1, device=device))
    optimizer = optim.Adam([dummy_param], lr=1e-4)

    # simulate Trainer logic: if D_t != D_s create teacher_proj and register
    if D_t != D_s:
        teacher_proj = nn.Linear(D_t, D_s).to(device)
        try:
            optimizer.add_param_group({'params': [p for p in teacher_proj.parameters() if p.requires_grad]})
        except Exception as e:
            print('failed to add param group:', e)
    else:
        teacher_proj = None

    print('teacher_proj created:', teacher_proj is not None)
    if teacher_proj is not None:
        print('teacher_proj weight shape:', tuple(teacher_proj.weight.shape))
        # check inclusion in optimizer
        found = False
        for g in optimizer.param_groups:
            for p in g['params']:
                if any(p is q for q in teacher_proj.parameters()):
                    found = True
        print('teacher_proj in optimizer param_groups:', found)

    # project teacher embeddings and compute cosine loss with student
    if teacher_proj is not None:
        t_proj = teacher_proj(teacher_embs)  # B x L x D_s
    else:
        t_proj = teacher_embs

    # compute student mask via prototypes (simulate prototypes with random tensors)
    # create fake prototypes matching D_s
    proto_img = torch.randn(32, D_s, device=student_embs.device)
    proto_mask = torch.randn(32, D_s, device=student_embs.device)
    proto_img = F.normalize(proto_img, dim=1)
    proto_mask = F.normalize(proto_mask, dim=1)

    s_mask, weights = StudentMaskComputer.compute(student_embs, proto_img, proto_mask, temp=0.1)

    s_norm = F.normalize(student_embs, dim=2)
    t_norm = F.normalize(t_proj, dim=2)
    s_flat = s_norm.view(-1, D_s)
    t_flat = t_norm.view(-1, D_s)
    loss_img = 1.0 - (s_flat * t_flat).sum(dim=1).mean()
    print('simulated image distill loss:', float(loss_img))
    # simulated mask loss against projected teacher if existed
    # for simulation, use t_proj as teacher mask proxy
    sm = s_mask.view(-1, D_s)
    tm = t_norm.view(-1, D_s)
    loss_mask = 1.0 - (sm * tm).sum(dim=1).mean()
    print('simulated mask distill loss:', float(loss_mask))


if __name__ == '__main__':
    simulate_projection_creation()

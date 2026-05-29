import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from functools import partial
from tqdm import tqdm
from models.losses import CharbonnierLoss


def timestep_embedding(timesteps, dim, max_period=10000):
    """Create sinusoidal embeddings for timesteps.

    timesteps: 1D LongTensor of shape (B,)
    dim: embedding dimension
    returns: FloatTensor shape (B, dim)
    """
    assert timesteps.dim() == 1
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, device=timesteps.device).float() / float(half))
    args = timesteps.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros(timesteps.shape[0], 1, device=timesteps.device)], dim=1)
    return emb


class SemanticHead(nn.Module):
    """轻量语义头：把重建的 RGB 图像映射为语义图（可放在 DDPM/ DDIM 中）"""
    def __init__(self, semantic_dim=32, mid_channels=64, in_channels=3, with_sigmoid=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(mid_channels, semantic_dim, 3, padding=1)
        ]
        if with_sigmoid:
            layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, img):
        return self.net(img)

def extract(a, t, x_shape):
    # Ensure timestep indices are integer type on the same device as `a`
    t_idx = t.to(dtype=torch.long, device=a.device)
    b, *_ = t_idx.shape
    out = a.gather(-1, t_idx)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

class DDPM(nn.Module):
    def __init__(self, model, img_channels, betas, criterion='l1', device='cuda', semantic_dim=32, semantic_mid_channels=64):
        super().__init__()
        self.model = nn.DataParallel(model).to(device)
        self.img_channels = img_channels
        # 语义头放到 diffusion wrapper 里，基于 pred_x0 进行语义预测
        self.semantic_dim = semantic_dim
        # Create semantic head to output logits by default (no sigmoid).
        # semantic head operates on predicted x0 (RGB), so use 3 input channels.
        self.semantic_head = SemanticHead(self.semantic_dim, mid_channels=semantic_mid_channels, in_channels=3, with_sigmoid=False).to(device)
        self.num_timesteps = len(betas)
        if criterion == 'l1':
            self.criterion = CharbonnierLoss()
        elif criterion == 'l2':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError("loss criterion must be l1 or l2")
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas", to_torch(alphas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))

        self.register_buffer("sqrt_alphas_cumprod",
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod",
                             to_torch(np.sqrt(1 - alphas_cumprod)))
        self.register_buffer("reciprocal_sqrt_alphas",
                             to_torch(np.sqrt(1 / alphas)))

        self.register_buffer("remove_noise_coeff", to_torch(
            betas / np.sqrt(1 - alphas_cumprod)))
        self.register_buffer("sigma", to_torch(np.sqrt(betas)))

    @torch.no_grad()
    def remove_noise(self, x, condition, t):
        # condition may be either a single spatial tensor (B x C_cond x H x W)
        # or a tuple/list (lr, cond_map). Normalize to full spatial condition.
        if isinstance(condition, (tuple, list)) and len(condition) == 2:
            lr, cond_map = condition
            cond_spatial = torch.cat([lr, cond_map], dim=1)
        else:
            cond_spatial = condition
        input = torch.cat([x, cond_spatial], dim=1)
        # convert timestep indices to embedding vector expected by UNet
        t_idx = t.to(dtype=torch.long, device=x.device)
        # determine base_channels from wrapped model if DataParallel
        try:
            base_ch = self.model.module.base_channels
        except Exception:
            base_ch = self.model.base_channels
        t_vec = timestep_embedding(t_idx.view(-1), base_ch)
        return ((x - extract(self.remove_noise_coeff, t_idx, x.shape) * self.model(input, t_vec)) * extract(self.reciprocal_sqrt_alphas, t_idx, x.shape))

    @torch.no_grad()
    def sample(self, condition, device, tqdm_visible=False):
        # condition may be (lr, cond_map) or a spatial tensor
        if isinstance(condition, (tuple, list)) and len(condition) == 2:
            lr, cond_map = condition
            full_condition = torch.cat([lr, cond_map], dim=1)
        else:
            full_condition = condition
        b, c, h, w = full_condition.shape
        x = torch.randn((b, 3, h, w), device=device)

        if tqdm_visible:
            timesteps_list = tqdm(range(self.num_timesteps - 1, -1, -1), desc='sampling loop time step', total=self.num_timesteps)
        else:
            timesteps_list = range(self.num_timesteps - 1, -1, -1)

        for t in timesteps_list:
            t_batch = torch.tensor([t], device=device).repeat(b)
            x = self.remove_noise(x, condition, t_batch)

            if t > 0:
                x += extract(self.sigma, t_batch, x.shape) * \
                    torch.randn_like(x)

        return x.cpu().detach()

    def perturb_x(self, x, t, noise):
        return (extract(self.sqrt_alphas_cumprod, t, x.shape) * x + extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * noise)

    def compute_loss(self, x, condition, t):
        # Ensure we have integer indices for buffer lookups, but pass a float
        # timestep to the model if it expects float embeddings.
        t_idx = t.to(dtype=torch.long, device=x.device)
        noise = torch.randn_like(x)
        pred = self.perturb_x(x, t_idx, noise)  # x_t (noisy)
        # condition may be (lr, cond_map) or a full spatial condition
        if isinstance(condition, (tuple, list)) and len(condition) == 2:
            lr, cond_map = condition
            cond_spatial = torch.cat([lr, cond_map], dim=1)
        else:
            cond_spatial = condition
        input_noisy = torch.cat([pred, cond_spatial], dim=1)

        # Build a timestep vector embedding matching UNet's expected input
        try:
            base_ch = self.model.module.base_channels
        except Exception:
            base_ch = self.model.base_channels
        t_vec = timestep_embedding(t_idx.view(-1), base_ch)

        # 模型可能返回单个张量（预测噪声）或 (pred_noise, semantic_out)
        model_out = self.model(input_noisy, t_vec)
        if isinstance(model_out, (tuple, list)):
            pred_noise = model_out[0]
        else:
            pred_noise = model_out

        loss = self.criterion(pred_noise, noise)

        # 估计 x0（对 clean image 的预测），用于后续语义/感知监督
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t_idx, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t_idx, x.shape)
        pred_x0 = (pred - sqrt_one_minus_alphas_cumprod_t * pred_noise) / sqrt_alphas_cumprod_t

        # 基于 pred_x0 获得语义输出（使用 wrapper 内的 semantic_head）
        semantic_out = None
        try:
            semantic_out = self.semantic_head(pred_x0)
        except Exception:
            semantic_out = None

        # 返回 loss, semantic_out, pred_x0（兼容上层处理）
        return loss, semantic_out, pred_x0

    def forward(self, x, condition):
        b, c, h, w = x.shape
        device = x.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device)
        return self.compute_loss(x, condition, t)



class DDIM(DDPM):
    @torch.no_grad()
    def sample(self, condition, patch_emb=None, sample_timesteps=20, ddim_eta=0.0, device="cuda", tqdm_visible=False, init_noise=None):
        # Allow `condition` to be either a spatial tensor (B x C x H x W)
        # or a tuple/list `(lr, cond_map)` similar to other methods.
        if isinstance(condition, (tuple, list)) and len(condition) == 2:
            lr, cond_map = condition
            cond_spatial = torch.cat([lr, cond_map], dim=1)
        else:
            cond_spatial = condition

        b, c, h, w = cond_spatial.shape

        # 构建采样时间序列
        ddim_timestep_seq = np.asarray(
            list(range(0, self.num_timesteps, self.num_timesteps // sample_timesteps))
        ) + 1
        ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq[:-1])

        # 初始化噪声
        if init_noise is not None:
            x = init_noise
        else:
            x = torch.randn((b, 3, h, w), device=device)

        # 采样循环
        if tqdm_visible:
            timesteps_list = tqdm(reversed(range(0, sample_timesteps)), desc='sampling loop time step', total=sample_timesteps)
        else:
            timesteps_list = reversed(range(0, sample_timesteps))

        semantic_outputs = []

        for i in timesteps_list:
            t_batch = torch.tensor(
                [ddim_timestep_seq[i]], device=device, dtype=torch.long
            ).repeat(b)
            prev_t_batch = torch.tensor(
                [ddim_timestep_prev_seq[i]], device=device, dtype=torch.long
            ).repeat(b)

            # 1. get current and previous alpha_cumprod
            alpha_cumprod_t = extract(self.alphas_cumprod, t_batch, x.shape)
            alpha_cumprod_t_prev = extract(self.alphas_cumprod, prev_t_batch, x.shape)


            # 2. UNet forward -> 预测噪声（model 返回 sr_out，即噪声预测）
            # Build time embedding vector consistent with compute_loss/remove_noise
            try:
                base_ch = self.model.module.base_channels
            except Exception:
                base_ch = self.model.base_channels
            t_vec = timestep_embedding(t_batch.view(-1), base_ch)
            # Build model input by concatenating current x and spatial condition (consistent with compute_loss)
            # If patch_emb is provided (non-spatial), we ignore it for now and use the spatial condition.
            model_in = torch.cat([x, cond_spatial], dim=1)
            model_out = self.model(model_in, t_vec)
            if isinstance(model_out, (tuple, list)):
                sr_out = model_out[0]
            else:
                sr_out = model_out

            # 3. 预测噪声
            pred_noise = sr_out  # UNet 输出预测噪声

            # 4. compute predicted x0
            pred_x0 = (x - torch.sqrt(1. - alpha_cumprod_t) * pred_noise) / torch.sqrt(alpha_cumprod_t)

            # 5. 基于 pred_x0 计算语义输出（使用 wrapper 内的 semantic_head）
            try:
                semantic_out = self.semantic_head(pred_x0)
            except Exception:
                semantic_out = None

            semantic_outputs.append(semantic_out)
            # 6. compute variance: "sigma_t(η)"
            sigmas_t = ddim_eta * torch.sqrt(
                (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev)
            )

            # 7. compute direction pointing to x_t
            pred_dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - sigmas_t**2) * pred_noise

            # 8. compute x_{t-1}
            x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir_xt + sigmas_t * torch.randn_like(x)
            x = x_prev

        # 返回最终 SR 图像和所有时间步的 semantic 输出
        semantic_outputs = torch.stack(semantic_outputs, dim=1)  # b x T x C x H x W
        return x.cpu().detach(), semantic_outputs.cpu().detach()



class EMA():
    def __init__(self, decay):
        self.decay = decay

    def __call__(self, old, new):
        old_dict = old.state_dict()
        new_dict = new.state_dict()
        for key in old_dict.keys():
            new_dict[key].data = old_dict[key].data * \
                self.decay + new_dict[key].data * (1 - self.decay)
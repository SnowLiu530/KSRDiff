import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------- 基础模块 -----------------
class TimeEmbedding(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fc = nn.Linear(channels, channels)
    def forward(self, t):
        return self.fc(t)

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.1, time_dim=None, activatedfun=nn.SiLU):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.act1 = activatedfun()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act2 = activatedfun()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.time_dim = time_dim
        if time_dim is not None:
            self.time_emb = nn.Linear(time_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb=None, cond_emb=None):
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)
        if t_emb is not None:
            h = h + self.time_emb(t_emb)[:, :, None, None]
        h = self.norm2(h)
        h = self.act2(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.skip(x)

class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AvgPool2d(2)
    def forward(self, x, *args, **kwargs):
        return self.pool(x)

class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
    def forward(self, x, *args, **kwargs):
        return self.up(x)

# ----------------- 改造后的 UNet -----------------
class UNet(nn.Module):
    def __init__(self, img_channels=9, base_channels=64, channel_mults=(1,2,3), 
                 num_res_blocks=2, time_dim=256, activatedfun=nn.SiLU, dropout=0.1,
                 semantic_dim=32):
        super().__init__()
        self.time_embedding = nn.Sequential(
            TimeEmbedding(base_channels),
            nn.Linear(base_channels, time_dim),
            activatedfun(),
            nn.Linear(time_dim, time_dim),
        )

        self.init_conv = nn.Conv2d(img_channels, base_channels, 3, padding=1)

        self.downblocks = nn.ModuleList()
        channels = [base_channels]
        now_channels = base_channels
        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResidualBlock(now_channels, out_channels, dropout, time_dim=time_dim, activatedfun=activatedfun))
                now_channels = out_channels
                channels.append(now_channels)
            if i != len(channel_mults) - 1:
                self.downblocks.append(Downsample(now_channels))
                channels.append(now_channels)

        self.mid = nn.ModuleList([
            ResidualBlock(now_channels, now_channels, dropout, time_dim=time_dim, activatedfun=activatedfun),
            ResidualBlock(now_channels, now_channels, dropout, time_dim=time_dim, activatedfun=activatedfun),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_channels = base_channels * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResidualBlock(channels.pop() + now_channels, out_channels, dropout, time_dim=time_dim, activatedfun=activatedfun))
                now_channels = out_channels
            if i != 0:
                self.upblocks.append(Upsample(now_channels))

        assert len(channels) == 0

        self.last_layer = nn.Sequential(
            activatedfun(),
            nn.Conv2d(base_channels, 3, 3, padding=1)
        )

        # 保存 base_channels 以便后续使用
        self.base_channels = base_channels

    def forward(self, x, time, cond_emb=None):
        """
        x: LR + 其他条件输入 (B, C, H, W)
        time: diffusion time embedding (B, time_dim)
        cond_emb: cross-attention 条件 embedding (可选)
        """
        time_emb = self.time_embedding(time)
        x = self.init_conv(x)
        skips = [x]

        for layer in self.downblocks:
            if isinstance(layer, ResidualBlock):
                x = layer(x, time_emb, cond_emb)
            else:
                x = layer(x)

            skips.append(x)

        for layer in self.mid:
            x = layer(x, time_emb, cond_emb)

        for layer in self.upblocks:
            if isinstance(layer, ResidualBlock):
                x = torch.cat([x, skips.pop()], dim=1)
                x = layer(x, time_emb, cond_emb)
            else:
                x = layer(x)

        sr_out = self.last_layer(x)
        assert len(skips) == 0
        return sr_out

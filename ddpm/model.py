import math
import torch
from torch import nn
import torch.nn.functional as F

# Function to use to embed the timestep in order to give it as a condition to the model

def timestep_embedding(t, dim):
    # t: (B,) integer timesteps  ->  (B, dim) float
    half = dim // 2
    # (half,)  geometric, 1 down to 1/10000
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    # (B, half)
    args = t[:, None] * freqs[None, :]
    # (B, dim)
    return torch.cat([args.sin(), args.cos()], dim=1)

# The block we will use in the UNet

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, temb_dim):
        super().__init__()
        # GroupNorm, 8 groups, over in_ch
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=in_ch)
        # 3x3, in_ch -> out_ch, padding 1
        self.conv1  = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        # Linear, temb_dim -> out_ch
        self.time_proj = nn.Linear(temb_dim, out_ch)
        # GroupNorm, 8 groups, over out_ch
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=out_ch)
        # 3x3, out_ch -> out_ch, padding 1
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        # 1x1 conv if channels change, else nn.Identity()
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, temb):
        # conv1( silu( norm1(x) ) )
        h = self.conv1(F.silu(self.norm1(x)))
        # h + time_proj( silu(temb) ) reshaped to (B, out_ch, 1, 1)
        h = h + self.time_proj(F.silu(temb))[:, :, None, None]
        # conv2( silu( norm2(h) ) )
        h = self.conv2(F.silu(self.norm2(h)))
        # h + skip(x)
        return h + self.skip(x)

# The Attention Block that will be used on the bottleneck of the UNet

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        # GroupNorm, 8 groups, over channels
        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)

        # Projections W_q, W_k, W_v
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        
        self.output_proj = nn.Linear(channels, channels)

        # Set the weights and bias of output_proj to zero 
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x):
        # input x : (B, C, H, W)
        # Normalize
        x_normalized = self.norm(x) # (B, C, H, W)
        # Flatten
        x_flat = torch.flatten(x_normalized, start_dim=-2) # (B, C, N), where N = H x W
        # Swap axes and get the tokens we will use
        tokens = torch.swapaxes(x_flat, -1, -2) # (B, N, C)

        # Apply the projections to tokens, and get q, k, v.
        q = self.q_proj(tokens) # (B, N, C)
        k = self.k_proj(tokens) # (B, N, C)
        v = self.v_proj(tokens) # (B, N, C)

        scores = (q @ torch.swapaxes(k, -1, -2)) / (self.channels ** 0.5) # (B, N, N)
        weights = F.softmax(scores, dim=2) # (B, N, N)
        attn_out = weights @ v # (B, N, C)
        projected = self.output_proj(attn_out) # (B, N, C)
        swapped_back = torch.swapaxes(projected, -1, -2) # (B, C, N)
        unflattened = swapped_back.reshape(x.shape) # (B, C, H, W)

        return x + unflattened

# UNet, which is the keystone (model) and the action will happen

class UNet(nn.Module):
    def __init__(self, in_ch=1, base=64, temb_dim=256, attention=True):
        super().__init__()
        # forward needs this to call timestep_embedding
        self.temb_dim = temb_dim

        # Linear -> SiLU -> Linear, all temb_dim
        self.temb_mlp = nn.Sequential(
            nn.Linear(self.temb_dim, self.temb_dim),
            nn.SiLU(),
            nn.Linear(self.temb_dim, self.temb_dim)
        )

        # conv 1 -> 64
        self.stem = nn.Conv2d(in_ch, base, 3, padding=1)

        # ResBlocks @ 32x32
        self.d1a, self.d1b = ResBlock(base, base, temb_dim), ResBlock(base, base, temb_dim)

        # stride-2 conv, 64 -> 64
        self.down1 = nn.Conv2d(base, base, kernel_size=3, stride=2, padding=1)

        # ResBlocks @ 16x16
        self.d2a, self.d2b = ResBlock(base, base*2, temb_dim), ResBlock(base*2, base*2, temb_dim)

        # stride-2 conv, 128 -> 128
        self.down2 = nn.Conv2d(base*2, base*2, kernel_size=3, stride=2, padding=1)

        # ResBlock @ 8x8
        self.mid = ResBlock(base*2, base*2, temb_dim)

        # AttentionBlock 
        self.attn = AttentionBlock(base*2) if attention else nn.Identity()

        # Upsample + conv, -> 16x16
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base*2, base*2, kernel_size=3, padding=1)
        )

        # ResBlocks after concat (in_ch = 256)
        self.u1a, self.u1b = ResBlock(base*4, base*2, temb_dim), ResBlock(base*2, base*2, temb_dim)

        # Upsample + conv, -> 32x32
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base*2, base*2, kernel_size=3, padding=1)
        )

        # ResBlocks after concat (in_ch = 192)
        self.u2a, self.u2b = ResBlock(base*3, base, temb_dim), ResBlock(base, base, temb_dim)

        # GroupNorm(8, 64)
        self.out_norm = nn.GroupNorm(8, base)

        # conv 64 -> 1
        self.out_conv = nn.Conv2d(base, in_ch, kernel_size=3, padding=1)

    def forward(self, x, t):
        # timestep_embedding -> temb_mlp, ONCE
        temb = self.temb_mlp(timestep_embedding(t, self.temb_dim))

        # stem
        h = self.stem(x)

        # d1a
        h = self.d1a(h, temb)
        # d1b
        h = self.d1b(h, temb)
        
        # stash
        A = h

        # down1
        h = self.down1(h)

        # d2a
        h = self.d2a(h, temb)
        # d2b
        h = self.d2b(h, temb)
        
        # stash
        B = h

        # down2
        h = self.down2(h)
        
        # mid
        h = self.mid(h, temb)

        # attention
        h = self.attn(h)

        # up1
        h = self.up1(h)
        
        # cat with B
        h = torch.cat([h, B], dim=1)

        # u1a
        h = self.u1a(h, temb)
        # u1b
        h = self.u1b(h, temb)

        # up2
        h = self.up2(h)
        
        # cat with A
        h = torch.cat([h, A], dim=1)

        # u2a
        h = self.u2a(h, temb)
        
        # u2b
        h = self.u2b(h, temb)

        # out_conv(silu(out_norm(h)))
        return self.out_conv(F.silu(self.out_norm(h)))


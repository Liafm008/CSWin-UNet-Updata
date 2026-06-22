
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from timm.layers import DropPath, trunc_normal_
except ImportError:
    from timm.models.layers import DropPath, trunc_normal_
from einops.layers.torch import Rearrange
import torch.utils.checkpoint as checkpoint
import numpy as np


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LePEAttention(nn.Module):
    def __init__(self, dim, resolution, idx, split_size, dim_out=None, num_heads=9, attn_drop=0., proj_drop=0.,
                 qk_scale=None):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        if idx == -1:
            H_sp, W_sp = self.resolution, self.resolution
        elif idx == 0:
            H_sp, W_sp = self.resolution, self.split_size
        elif idx == 1:
            W_sp, H_sp = self.resolution, self.split_size
        else:
            print("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp
        stride = 1
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.attn_drop = nn.Dropout(attn_drop)

    def im2cswin(self, x):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp * self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def get_lepe(self, x, func):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)

        H_sp, W_sp = self.H_sp, self.W_sp
        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp)  ### B', C, H', W'

        lepe = func(x)  ### B', C, H', W'
        lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()

        x = x.reshape(-1, self.num_heads, C // self.num_heads, self.H_sp * self.W_sp).permute(0, 1, 3, 2).contiguous()
        return x, lepe

    def forward(self, qkv):
        """
        x: B L C
        """
        q, k, v = qkv[0], qkv[1], qkv[2]

        ### Img2Window
        H = W = self.resolution
        B, L, C = q.shape

        assert L == H * W, "flatten img_tokens has wrong size"

        q = self.im2cswin(q)
        k = self.im2cswin(k)
        v, lepe = self.get_lepe(v, self.get_v)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v) + lepe
        x = x.transpose(1, 2).reshape(-1, self.H_sp * self.W_sp, C)  # B head N N @ B head N C

        ### Window2Img
        x = windows2img(x, self.H_sp, self.W_sp, H, W).view(B, -1, C)  # B H' W' C

        return x


class CSWinBlock(nn.Module):

    def __init__(self, dim, reso, num_heads,
                 split_size, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 last_stage=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patches_resolution = reso
        self.split_size = split_size
        self.mlp_ratio = mlp_ratio
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm1 = norm_layer(dim)

        if self.patches_resolution == split_size:
            last_stage = True
        if last_stage:
            self.branch_num = 1
        else:
            self.branch_num = 2
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        if last_stage:
            self.attns = nn.ModuleList([
                LePEAttention(
                    dim, resolution=self.patches_resolution, idx=-1,
                    split_size=split_size, num_heads=num_heads, dim_out=dim,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                for i in range(self.branch_num)])

        else:
            self.attns = nn.ModuleList([
                LePEAttention(
                    dim // 2, resolution=self.patches_resolution, idx=i,
                    split_size=split_size, num_heads=num_heads // 2, dim_out=dim // 2,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                for i in range(self.branch_num)])

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                       drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """

        H = W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"
        img = self.norm1(x)
        qkv = self.qkv(img).reshape(B, -1, 3, C).permute(2, 0, 1, 3)

        if self.branch_num == 2:
            x1 = self.attns[0](qkv[:, :, :, :C // 2])
            x2 = self.attns[1](qkv[:, :, :, C // 2:])
            attened_x = torch.cat([x1, x2], dim=2)
        else:
            attened_x = self.attns[0](qkv)
        attened_x = self.proj(attened_x)
        x = x + self.drop_path(attened_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


def img2windows(img, H_sp, W_sp):
    """
    img: B C H W
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp * W_sp, C)
    return img_perm


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    img_splits_hw: B' H W C
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img


class Merge_Block(nn.Module):
    def __init__(self, dim, dim_out, norm_layer=nn.LayerNorm):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim_out, 3, 2, 1)
        self.norm = norm_layer(dim_out)

    def forward(self, x):
        B, new_HW, C = x.shape
        H = W = int(np.sqrt(new_HW))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = self.conv(x)
        B, C = x.shape[:2]
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()
        x = self.norm(x)

        return x

class CARAFE(nn.Module):
    def __init__(self, dim, dim_out, kernel_size=3, up_factor=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.up_factor = up_factor
        self.down = nn.Conv2d(dim, dim // 4, 1)
        self.encoder = nn.Conv2d(dim // 4, self.up_factor ** 2 * self.kernel_size ** 2,
                                 self.kernel_size, 1, self.kernel_size // 2)
        self.out = nn.Conv2d(dim, dim_out, 1)

    def forward(self, x):
        B, new_HW, C = x.shape
        H = W = int(np.sqrt(new_HW))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)


            # N,C,H,W -> N,C,delta*H,delta*W
            # kernel prediction module
        kernel_tensor = self.down(x)  # (N, Cm, H, W)
        kernel_tensor = self.encoder(kernel_tensor)  # (N, S^2 * Kup^2, H, W)
        kernel_tensor = F.pixel_shuffle(kernel_tensor,
                                        self.up_factor)  # (N, S^2 * Kup^2, H, W)->(N, Kup^2, S*H, S*W)
        kernel_tensor = F.softmax(kernel_tensor, dim=1)  # (N, Kup^2, S*H, S*W)
        kernel_tensor = kernel_tensor.unfold(2, self.up_factor, step=self.up_factor)  # (N, Kup^2, H, W*S, S)
        kernel_tensor = kernel_tensor.unfold(3, self.up_factor, step=self.up_factor)  # (N, Kup^2, H, W, S, S)
        kernel_tensor = kernel_tensor.reshape(B, self.kernel_size ** 2, H, W,
                                                  self.up_factor ** 2)  # (N, Kup^2, H, W, S^2)
        kernel_tensor = kernel_tensor.permute(0, 2, 3, 1, 4)  # (N, H, W, Kup^2, S^2)

            # content-aware reassembly module
            # tensor.unfold: dim, size, step
        w = F.pad(x, pad=(self.kernel_size // 2, self.kernel_size // 2,
                                              self.kernel_size // 2, self.kernel_size // 2),
                              mode='constant', value=0)  # (N, C, H+Kup//2+Kup//2, W+Kup//2+Kup//2)
        w = w.unfold(2, self.kernel_size, step=1)  # (N, C, H, W+Kup//2+Kup//2, Kup)
        w = w.unfold(3, self.kernel_size, step=1)  # (N, C, H, W, Kup, Kup)
        w = w.reshape(B, C, H, W, -1)  # (N, C, H, W, Kup^2)
        w = w.permute(0, 2, 3, 1, 4)  # (N, H, W, C, Kup^2)

        x = torch.matmul(w, kernel_tensor)  # (N, H, W, C, S^2)
        x = x.reshape(B, H, W, -1)
        x = x.permute(0, 3, 1, 2)
        x = F.pixel_shuffle(x, self.up_factor)
        x = self.out(x)
        B, C = x.shape[:2]
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()

        return x


class CARAFE4(nn.Module):
    def __init__(self, dim, dim_out, kernel_size=3, up_factor=4):
        super().__init__()
        self.kernel_size = kernel_size
        self.up_factor = up_factor
        self.down = nn.Conv2d(dim, dim // 4, 1)
        self.encoder = nn.Conv2d(dim // 4, self.up_factor ** 2 * self.kernel_size ** 2,
                                 self.kernel_size, 1, self.kernel_size // 2)
        self.out = nn.Conv2d(dim, dim_out, 1)

    def forward(self, x):
        B, new_HW, C = x.shape
        H = W = int(np.sqrt(new_HW))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)


            # N,C,H,W -> N,C,delta*H,delta*W
            # kernel prediction module
        kernel_tensor = self.down(x)  # (N, Cm, H, W)
        kernel_tensor = self.encoder(kernel_tensor)  # (N, S^2 * Kup^2, H, W)
        kernel_tensor = F.pixel_shuffle(kernel_tensor,
                                        self.up_factor)  # (N, S^2 * Kup^2, H, W)->(N, Kup^2, S*H, S*W)
        kernel_tensor = F.softmax(kernel_tensor, dim=1)  # (N, Kup^2, S*H, S*W)
        kernel_tensor = kernel_tensor.unfold(2, self.up_factor, step=self.up_factor)  # (N, Kup^2, H, W*S, S)
        kernel_tensor = kernel_tensor.unfold(3, self.up_factor, step=self.up_factor)  # (N, Kup^2, H, W, S, S)
        kernel_tensor = kernel_tensor.reshape(B, self.kernel_size ** 2, H, W,
                                                  self.up_factor ** 2)  # (N, Kup^2, H, W, S^2)
        kernel_tensor = kernel_tensor.permute(0, 2, 3, 1, 4)  # (N, H, W, Kup^2, S^2)

            # content-aware reassembly module
            # tensor.unfold: dim, size, step
        w = F.pad(x, pad=(self.kernel_size // 2, self.kernel_size // 2,
                                              self.kernel_size // 2, self.kernel_size // 2),
                              mode='constant', value=0)  # (N, C, H+Kup//2+Kup//2, W+Kup//2+Kup//2)
        w = w.unfold(2, self.kernel_size, step=1)  # (N, C, H, W+Kup//2+Kup//2, Kup)
        w = w.unfold(3, self.kernel_size, step=1)  # (N, C, H, W, Kup, Kup)
        w = w.reshape(B, C, H, W, -1)  # (N, C, H, W, Kup^2)
        w = w.permute(0, 2, 3, 1, 4)  # (N, H, W, C, Kup^2)

        x = torch.matmul(w, kernel_tensor)  # (N, H, W, C, S^2)
        x = x.reshape(B, H, W, -1)
        x = x.permute(0, 3, 1, 2)
        x = F.pixel_shuffle(x, self.up_factor)
        x = self.out(x)
        B, C = x.shape[:2]
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()

        return x


def _tokens_to_feature_map(x):
    B, new_HW, C = x.shape
    H = W = int(np.sqrt(new_HW))
    if H * W != new_HW:
        raise ValueError("feature tokens must describe a square feature map")
    return x.transpose(-2, -1).contiguous().view(B, C, H, W)


def _feature_map_to_tokens(x):
    B, C = x.shape[:2]
    return x.contiguous().view(B, C, -1).transpose(-2, -1).contiguous()


class SkipAttentionBlock(nn.Module):
    """CBAM-style spatial and channel attention for one skip feature."""

    def __init__(self, dim, reduction=16):
        super().__init__()
        hidden_dim = max(dim // reduction, 4)
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False),
        )
        self.channel_gate = nn.Sigmoid()

    def forward_map(self, x):
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map = torch.amax(x, dim=1, keepdim=True)
        x = x * self.spatial(torch.cat([avg_map, max_map], dim=1))

        avg_pool = F.adaptive_avg_pool2d(x, 1)
        max_pool = F.adaptive_max_pool2d(x, 1)
        channel_weight = self.channel_gate(
            self.channel_mlp(avg_pool) + self.channel_mlp(max_pool)
        )
        return x * channel_weight

    def forward(self, x):
        return _feature_map_to_tokens(self.forward_map(_tokens_to_feature_map(x)))


class SkipAttentionFusion(nn.Module):
    def __init__(self, dims, init_scale=0.5):
        super().__init__()
        self.blocks = nn.ModuleList([SkipAttentionBlock(dim) for dim in dims])
        self.scales = nn.Parameter(torch.full((len(dims),), float(init_scale)))

    def forward(self, features):
        refined = []
        for i, feature in enumerate(features[:len(self.blocks)]):
            attended = self.blocks[i](feature)
            scale = self.scales[i].to(device=feature.device, dtype=feature.dtype)
            refined.append(feature + scale * (attended - feature))
        return tuple(refined)


class DecoderGuidedSkipGate(nn.Module):
    """Use decoder semantics to modulate an encoder skip before concatenation."""

    def __init__(self, skip_dim, decoder_dim, inter_dim=None, init_scale=0.1):
        super().__init__()
        inter_dim = inter_dim or max(skip_dim // 4, 16)
        self.skip_proj = nn.Conv2d(skip_dim, inter_dim, kernel_size=1, bias=False)
        self.decoder_proj = nn.Conv2d(
            decoder_dim, inter_dim, kernel_size=1, bias=False
        )
        self.norm = nn.GroupNorm(1, inter_dim)
        self.act = nn.GELU()
        self.gate = nn.Conv2d(inter_dim, 1, kernel_size=1)
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))

        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, skip, decoder):
        skip_map = _tokens_to_feature_map(skip)
        decoder_map = _tokens_to_feature_map(decoder)
        if decoder_map.shape[-2:] != skip_map.shape[-2:]:
            decoder_map = F.interpolate(
                decoder_map,
                size=skip_map.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        gate = self.skip_proj(skip_map) + self.decoder_proj(decoder_map)
        gate = torch.sigmoid(self.gate(self.act(self.norm(gate))))
        scale = self.scale.to(device=skip_map.device, dtype=skip_map.dtype)
        modulation = 1.0 + scale * (2.0 * gate - 1.0)
        return _feature_map_to_tokens(skip_map * modulation)


class ASPChannelAttentionBlock(nn.Module):
    """Residual channel attention inspired by ASP-VMUNet CAB."""

    def __init__(self, dim, kernel_size=3):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        attn = self.conv(attn).transpose(-1, -2).unsqueeze(-1)
        attn = self.sigmoid(attn)
        return x * attn + x


class ASPSpatialAttentionBlock(nn.Module):
    """Residual spatial attention inspired by ASP-VMUNet SAB."""

    def __init__(self, kernel_size=7, dilation=3):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map = torch.amax(x, dim=1, keepdim=True)
        attn = self.sigmoid(self.conv(torch.cat([avg_map, max_map], dim=1)))
        return x * attn + x


class SABCabBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.cab = ASPChannelAttentionBlock(dim)
        self.sab = ASPSpatialAttentionBlock()

    def forward(self, x):
        return self.sab(self.cab(x))


class SABCabFusion(nn.Module):
    def __init__(self, dims, init_scale=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([SABCabBlock(dim) for dim in dims])
        self.scales = nn.Parameter(torch.full((len(dims),), float(init_scale)))

    def forward(self, features):
        refined = []
        for i, feature in enumerate(features[:len(self.blocks)]):
            feature_map = _tokens_to_feature_map(feature)
            attended = _feature_map_to_tokens(self.blocks[i](feature_map))
            scale = self.scales[i].to(device=feature.device, dtype=feature.dtype)
            refined.append(feature + scale * (attended - feature))
        return tuple(refined)


class DepthwiseTokenProjection(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim, kernel_size=1, groups=dim, bias=False)

    def forward(self, x):
        return self.proj(x.transpose(1, 2)).transpose(1, 2).contiguous()


class ChannelCrossAttention(nn.Module):
    def __init__(self, dims):
        super().__init__()
        total_dim = sum(dims)
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for dim in dims])
        self.q_projs = nn.ModuleList([DepthwiseTokenProjection(dim) for dim in dims])
        self.out_projs = nn.ModuleList([DepthwiseTokenProjection(dim) for dim in dims])
        self.k_proj = DepthwiseTokenProjection(total_dim)
        self.v_proj = DepthwiseTokenProjection(total_dim)

    def forward(self, tokens):
        norm_tokens = [norm(token) for norm, token in zip(self.norms, tokens)]
        context = torch.cat(norm_tokens, dim=-1)
        key = self.k_proj(context).transpose(1, 2)
        value = self.v_proj(context).transpose(1, 2)
        scale = tokens[0].shape[1] ** -0.5

        outputs = []
        for token, query_proj, out_proj in zip(norm_tokens, self.q_projs, self.out_projs):
            query = query_proj(token).transpose(1, 2)
            attn = torch.softmax(query @ key.transpose(-2, -1) * scale, dim=-1)
            out = (attn @ value).transpose(1, 2).contiguous()
            outputs.append(out_proj(out))
        return outputs


class SpatialCrossAttention(nn.Module):
    def __init__(self, dims, num_heads=4):
        super().__init__()
        total_dim = sum(dims)
        if total_dim % num_heads != 0:
            raise ValueError("total DCA channels must be divisible by num_heads")
        self.num_heads = num_heads
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for dim in dims])
        self.q_proj = DepthwiseTokenProjection(total_dim)
        self.k_proj = DepthwiseTokenProjection(total_dim)
        self.v_projs = nn.ModuleList([DepthwiseTokenProjection(dim) for dim in dims])
        self.out_projs = nn.ModuleList([DepthwiseTokenProjection(dim) for dim in dims])

    def _split_heads(self, x):
        B, P, C = x.shape
        head_dim = C // self.num_heads
        return x.view(B, P, self.num_heads, head_dim).permute(0, 2, 1, 3).contiguous()

    def forward(self, tokens):
        norm_tokens = [norm(token) for norm, token in zip(self.norms, tokens)]
        context = torch.cat(norm_tokens, dim=-1)
        query = self._split_heads(self.q_proj(context))
        key = self._split_heads(self.k_proj(context))
        scale = query.shape[-1] ** -0.5
        attn = torch.softmax(query @ key.transpose(-2, -1) * scale, dim=-1)

        outputs = []
        for token, value_proj, out_proj in zip(norm_tokens, self.v_projs, self.out_projs):
            value = value_proj(token)
            B, P, C = value.shape
            value = value.unsqueeze(1).expand(B, self.num_heads, P, C)
            out = (attn @ value).mean(dim=1)
            outputs.append(out_proj(out))
        return outputs


class DCAFusion(nn.Module):
    """Dual Cross-Attention bridge for CSWin-UNet skip features."""

    def __init__(self, dims, init_scale=0.1, num_heads=4):
        super().__init__()
        self.cca = ChannelCrossAttention(dims)
        self.sca = SpatialCrossAttention(dims, num_heads=num_heads)
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for dim in dims])
        self.act = nn.GELU()
        self.scales = nn.Parameter(torch.full((len(dims),), float(init_scale)))

    def forward(self, features):
        feature_maps = [_tokens_to_feature_map(feature) for feature in features[:3]]
        target_size = feature_maps[-1].shape[-2:]

        pooled_tokens = []
        for feature_map in feature_maps:
            if feature_map.shape[-2:] == target_size:
                pooled = feature_map
            else:
                pooled = F.adaptive_avg_pool2d(feature_map, target_size)
            pooled_tokens.append(_feature_map_to_tokens(pooled))

        tokens = self.cca(pooled_tokens)
        tokens = self.sca(tokens)

        refined = []
        for i, (feature, feature_map, token) in enumerate(
            zip(features[:3], feature_maps, tokens)
        ):
            token = self.act(self.norms[i](token))
            dca_map = _tokens_to_feature_map(token)
            if dca_map.shape[-2:] != feature_map.shape[-2:]:
                dca_map = F.interpolate(
                    dca_map,
                    size=feature_map.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            dca_tokens = _feature_map_to_tokens(dca_map)
            scale = self.scales[i].to(device=feature.device, dtype=feature.dtype)
            refined.append(feature + scale * (dca_tokens - feature))
        return tuple(refined)


class SmoothConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class SDIFusion(nn.Module):
    """Semantics and Detail Infusion for CSWin-UNet skip features."""

    def __init__(
        self,
        dims,
        inter_channels=32,
        target_levels=3,
        init_scale=0.1,
        fusion_mode="product",
        use_attention=True,
        use_gate=False,
        target_indices=None,
    ):
        super().__init__()
        if target_indices is None:
            target_indices = list(range(target_levels))
        self.target_indices = list(target_indices)
        if not self.target_indices:
            raise ValueError("SDIFusion needs at least one target index")
        if min(self.target_indices) < 0 or max(self.target_indices) >= target_levels:
            raise ValueError(
                "SDIFusion target indices must be in the decoder skip range"
            )
        self.target_levels = len(self.target_indices)
        self.fusion_mode = fusion_mode
        self.use_attention = use_attention
        self.use_gate = use_gate
        self.attentions = (
            nn.ModuleList([SkipAttentionBlock(dim) for dim in dims])
            if use_attention
            else None
        )
        self.reductions = nn.ModuleList(
            [nn.Conv2d(dim, inter_channels, kernel_size=1) for dim in dims]
        )
        self.smooth = nn.ModuleList(
            [
                nn.ModuleList([SmoothConv(inter_channels) for _ in dims])
                for _ in range(self.target_levels)
            ]
        )
        self.fuse_norms = nn.ModuleList(
            [nn.GroupNorm(1, inter_channels) for _ in range(self.target_levels)]
        )
        self.projections = nn.ModuleList(
            [
                nn.Conv2d(inter_channels, dims[target_idx], kernel_size=1)
                for target_idx in self.target_indices
            ]
        )
        self.gates = (
            nn.ModuleList(
                [
                    nn.Conv2d(dims[target_idx], dims[target_idx], kernel_size=1)
                    for target_idx in self.target_indices
                ]
            )
            if use_gate
            else None
        )
        if self.gates is not None:
            for gate in self.gates:
                nn.init.zeros_(gate.weight)
                nn.init.constant_(gate.bias, -2.0)
        self.scales = nn.Parameter(
            torch.full((self.target_levels,), float(init_scale))
        )

    def forward(self, features):
        original_maps = []
        feature_maps = []
        for idx, (feature, reduction) in enumerate(zip(features, self.reductions)):
            feature_map = _tokens_to_feature_map(feature)
            original_maps.append(feature_map)
            if self.use_attention:
                feature_map = self.attentions[idx].forward_map(feature_map)
            feature_maps.append(reduction(feature_map))

        refined = list(features[:3])
        for level_idx, target_idx in enumerate(self.target_indices):
            target_size = feature_maps[target_idx].shape[-2:]
            resized_features = []
            for source_idx, feature_map in enumerate(feature_maps):
                if feature_map.shape[-2:] == target_size:
                    resized = feature_map
                elif feature_map.shape[-2] > target_size[0]:
                    resized = F.adaptive_avg_pool2d(feature_map, target_size)
                else:
                    resized = F.interpolate(
                        feature_map,
                        size=target_size,
                        mode="bilinear",
                        align_corners=False,
                    )
                resized_features.append(self.smooth[level_idx][source_idx](resized))

            fused = resized_features[0]
            if self.fusion_mode == "product":
                for resized in resized_features[1:]:
                    fused = fused * resized
            elif self.fusion_mode == "resprod":
                fused = resized_features[target_idx]
                for source_idx, resized in enumerate(resized_features):
                    if source_idx == target_idx:
                        continue
                    fused = fused * (1.0 + torch.tanh(resized))
            elif self.fusion_mode == "add":
                for resized in resized_features[1:]:
                    fused = fused + resized
                fused = fused / len(resized_features)
            else:
                raise ValueError("Unsupported SDI fusion mode: {}".format(self.fusion_mode))
            fused = self.fuse_norms[level_idx](fused)
            delta = self.projections[level_idx](fused)
            if self.use_gate:
                gate = torch.sigmoid(self.gates[level_idx](original_maps[target_idx]))
                delta = delta * gate
            delta = _feature_map_to_tokens(delta)

            feature = features[target_idx]
            scale = self.scales[level_idx].to(
                device=feature.device, dtype=feature.dtype
            )
            refined[target_idx] = feature + scale * delta
        return tuple(refined)


class CSWinTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=8, embed_dim=64, depth=[1, 2, 9, 1],
                 split_size=[1, 2, 7, 7],
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0, hybrid_backbone=None, norm_layer=nn.LayerNorm, use_chk=False,
                 skip_fusion="none", sdi_channels=32, skip_fusion_scale=0.1):
        super().__init__()
        self.use_chk = use_chk
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        heads = num_heads
        self.skip_fusion_type = skip_fusion

        #encoder

        self.stage1_conv_embed = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, 7, 4, 2),
            Rearrange('b c h w -> b (h w) c', h=img_size // 4, w=img_size // 4),
            nn.LayerNorm(embed_dim)
        )

        curr_dim = embed_dim

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, np.sum(depth))]  # stochastic depth decay rule
        self.stage1 = nn.ModuleList(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[0], reso=img_size // 4, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[0],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth[0])])
        self.merge1 = Merge_Block(curr_dim, curr_dim * 2)
        curr_dim = curr_dim * 2
        self.stage2 = nn.ModuleList(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[1], reso=img_size // 8, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[1],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:1]) + i], norm_layer=norm_layer)
                for i in range(depth[1])])
        self.merge2 = Merge_Block(curr_dim, curr_dim * 2)
        curr_dim = curr_dim * 2
        temp_stage3 = []
        temp_stage3.extend(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[2], reso=img_size // 16, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[2],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:2]) + i], norm_layer=norm_layer)
                for i in range(depth[2])])

        self.stage3 = nn.ModuleList(temp_stage3)
        self.merge3 = Merge_Block(curr_dim, curr_dim * 2)
        curr_dim = curr_dim * 2
        self.stage4 = nn.ModuleList(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[3], reso=img_size // 32, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[-1],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:-1]) + i], norm_layer=norm_layer, last_stage=True)
                for i in range(depth[-1])])

        self.norm = norm_layer(curr_dim)

        skip_dims = [embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8]
        self.decoder_skip_gates = None
        if skip_fusion == "none":
            self.skip_fusion = None
        elif skip_fusion == "decoder_gate":
            self.skip_fusion = None
            self.decoder_skip_gates = nn.ModuleList(
                [
                    DecoderGuidedSkipGate(
                        skip_dims[2], skip_dims[2], init_scale=skip_fusion_scale
                    ),
                    DecoderGuidedSkipGate(
                        skip_dims[1], skip_dims[1], init_scale=skip_fusion_scale
                    ),
                    DecoderGuidedSkipGate(
                        skip_dims[0], skip_dims[0], init_scale=skip_fusion_scale
                    ),
                ]
            )
        elif skip_fusion == "attention":
            self.skip_fusion = SkipAttentionFusion(
                skip_dims[:3], init_scale=skip_fusion_scale
            )
        elif skip_fusion == "sab_cab":
            self.skip_fusion = SABCabFusion(
                skip_dims[:3], init_scale=skip_fusion_scale
            )
        elif skip_fusion == "dca":
            self.skip_fusion = DCAFusion(
                skip_dims[:3], init_scale=skip_fusion_scale
            )
        elif skip_fusion == "sdi":
            self.skip_fusion = SDIFusion(
                skip_dims,
                inter_channels=sdi_channels,
                target_levels=3,
                init_scale=skip_fusion_scale,
            )
        elif skip_fusion == "sdi_mid":
            self.skip_fusion = SDIFusion(
                skip_dims,
                inter_channels=sdi_channels,
                target_levels=3,
                init_scale=skip_fusion_scale,
                target_indices=[1, 2],
            )
        elif skip_fusion == "sdi_resprod":
            self.skip_fusion = SDIFusion(
                skip_dims,
                inter_channels=sdi_channels,
                target_levels=3,
                init_scale=skip_fusion_scale,
                fusion_mode="resprod",
            )
        elif skip_fusion == "sdi_gate":
            self.skip_fusion = SDIFusion(
                skip_dims,
                inter_channels=sdi_channels,
                target_levels=3,
                init_scale=skip_fusion_scale,
                use_gate=True,
            )
        elif skip_fusion == "sdi_add":
            self.skip_fusion = SDIFusion(
                skip_dims,
                inter_channels=sdi_channels,
                target_levels=3,
                init_scale=skip_fusion_scale,
                fusion_mode="add",
                use_attention=False,
            )
        else:
            raise ValueError("Unsupported skip_fusion: {}".format(skip_fusion))

        # decoder


        self.stage_up4 = nn.ModuleList(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[3], reso=img_size // 32, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[-1],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:-1]) + i], norm_layer=norm_layer, last_stage=True)
                for i in range(depth[-1])])

        self.upsample4 = CARAFE(curr_dim, curr_dim // 2)
        curr_dim = curr_dim // 2

        self.concat_linear4 = nn.Linear(512, 256)
        self.stage_up3 = nn.ModuleList(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[2], reso=img_size // 16, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[2],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:2]) + i], norm_layer=norm_layer)
                for i in range(depth[2])]
        )

        self.upsample3 = CARAFE(curr_dim, curr_dim // 2)
        curr_dim = curr_dim // 2

        self.concat_linear3 = nn.Linear(256, 128)
        self.stage_up2 = nn.ModuleList(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[1], reso=img_size // 8, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[1],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:1]) + i], norm_layer=norm_layer)
                for i in range(depth[1])])
        self.upsample2 = CARAFE(curr_dim, curr_dim // 2)
        curr_dim = curr_dim // 2

        self.concat_linear2 = nn.Linear(128, 64)
        self.stage_up1 = nn.ModuleList([
            CSWinBlock(
                dim=curr_dim, num_heads=heads[0], reso=img_size // 4, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[0],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth[0])])

        self.upsample1 = CARAFE4(curr_dim, 64)
        self.norm_up = norm_layer(embed_dim)
        self.output = nn.Conv2d(in_channels=embed_dim, out_channels=self.num_classes, kernel_size=1, bias=False)
        # Classifier head

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def _checkpoint(self, block, x):
        return checkpoint.checkpoint(block, x, use_reentrant=False)

    #Encoder and Bottleneck
    def forward_features(self, x):
        x = self.stage1_conv_embed(x)

        x = self.pos_drop(x)

        for blk in self.stage1:
            if self.use_chk:
                x = self._checkpoint(blk, x)
            else:
                x = blk(x)
        self.x1 = x
        x = self.merge1(x)

        for blk in self.stage2:
            if self.use_chk:
                x = self._checkpoint(blk, x)
            else:
                x = blk(x)
        self.x2 = x
        x = self.merge2(x)

        for blk in self.stage3:
            if self.use_chk:
                x = self._checkpoint(blk, x)
            else:
                    x = blk(x)
        self.x3 = x
        x = self.merge3(x)

        for blk in self.stage4:
            if self.use_chk:
                x = self._checkpoint(blk, x)
            else:
                x = blk(x)

        x = self.norm(x)
        self.x4 = x

        if self.skip_fusion is not None:
            self.x1, self.x2, self.x3 = self.skip_fusion(
                (self.x1, self.x2, self.x3, self.x4)
            )

        return x

    #Dencoder and Skip connection
    def forward_up_features(self, x):
        for blk in self.stage_up4:
            if self.use_chk:
                x = self._checkpoint(blk, x)
            else:
                x = blk(x)
        x = self.upsample4(x)
        skip = self.x3
        if self.decoder_skip_gates is not None:
            skip = self.decoder_skip_gates[0](skip, x)
        x = torch.cat([skip, x],-1)
        x = self.concat_linear4(x)
        for blk in self.stage_up3:
            if self.use_chk:
                x = self._checkpoint(blk, x)
            else:
                x = blk(x)
        # print("decoder stage3", x.shape)
        x = self.upsample3(x)
        skip = self.x2
        if self.decoder_skip_gates is not None:
            skip = self.decoder_skip_gates[1](skip, x)
        x = torch.cat([skip, x],-1)
        x = self.concat_linear3(x)
        for blk in self.stage_up2:
            if self.use_chk:
                x = self._checkpoint(blk, x)
            else:
                    x = blk(x)
        x = self.upsample2(x)
        skip = self.x1
        if self.decoder_skip_gates is not None:
            skip = self.decoder_skip_gates[2](skip, x)
        x = torch.cat([skip, x],-1)
        x = self.concat_linear2(x)
        for blk in self.stage_up1:
            if self.use_chk:
                x = self._checkpoint(blk, x)
            else:
                x = blk(x)
        x = self.norm_up(x)  # B L C
        return x

    def up_x4(self, x):
        B, new_HW, C = x.shape
        H = W = int(np.sqrt(new_HW))
        x = self.upsample1(x)
        x = x.view(B, 4 * H, 4 * W, -1)
        x = x.permute(0, 3, 1, 2)  # B,C,H,W
        x = self.output(x)

        return x

    def forward(self, x):
        x = self.forward_features(x)

        x = self.forward_up_features(x)

        x = self.up_x4(x)


        return x

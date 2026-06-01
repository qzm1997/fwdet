# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Block modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad, Dconv
from .transformer import TransformerBlock
from ultralytics.utils.torch_utils import fuse_conv_and_bn
from pytorch_wavelets import DWTForward, DWTInverse
__all__ = (
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2fAttn",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "RepC3",
    "ResNetLayer",
    "RepNCSPELAN4",
    "ADown",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "Silence",
    "FSSB",
    "CARAFEPlus",
    "UpFuseBlockV2"
    "SplitFreq",
    "SobelConv",
    "FreqASFF"
)
class SobelConv(nn.Module):
    """
    SobelConv: 可学习的Sobel卷积层
    用于提取图像的边缘、轮廓等高维特征
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(SobelConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # 定义Sobel卷积核（水平和垂直方向）
        sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        # 将Sobel核扩展为可学习的卷积权重
        self.weight_x = nn.Parameter(sobel_kernel_x.view(1, 1, 3, 3).repeat(out_channels, in_channels, 1, 1))
        self.weight_y = nn.Parameter(sobel_kernel_y.view(1, 1, 3, 3).repeat(out_channels, in_channels, 1, 1))
        
        # 可选的缩放因子和偏置
        self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
        # 初始化
        with torch.no_grad():
            nn.init.kaiming_normal_(self.weight_x, mode='fan_out', nonlinearity='relu')
            nn.init.kaiming_normal_(self.weight_y, mode='fan_out', nonlinearity='relu')
        
    def forward(self, x):
        # 分别计算x方向和y方向的梯度
        grad_x = F.conv2d(x, self.weight_x, stride=self.stride, padding=self.padding)
        grad_y = F.conv2d(x, self.weight_y, stride=self.stride, padding=self.padding)
        
        # 合并梯度信息（梯度幅值）
        edge_feat = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
        
        # 缩放和偏置
        edge_feat = edge_feat * self.scale + self.bias.view(1, -1, 1, 1)
        
        return edge_feat
class ESCFFM(nn.Module):
    """
    YOLOv10兼容版本的Bottleneck_ESCFFM
    """
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3,3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        
        # 两条并行分支
        # 分支1：普通卷积（提取语义信息）
        self.conv_branch = nn.Sequential(
            Conv(c1, c_, 1, 1),
            Conv(c_, c_, k, 1, g=g, act=True)  # YOLOv10的Conv参数可能不同
        )
        
        # 分支2：SobelConv（提取边缘/高频信息）
        self.sobel_branch = nn.Sequential(
            Conv(c1, c_, 1, 1),
            Conv(c_, c_, 1, 1)  # 先用1x1降维
        )
        self.sobel_conv = SobelConv(c_, c_, k[0] if isinstance(k, tuple) else k, 1, padding=1)
        
        # 特征融合层
        self.fusion = Conv(c_ * 2, c2, 1, 1)
        
        self.add = shortcut and c1 == c2
        
    def forward(self, x):
        # 并行提取特征
        conv_out = self.conv_branch(x)
        
        # Sobel分支
        sobel_feat = self.sobel_branch(x)
        sobel_out = self.sobel_conv(sobel_feat)
        
        # 确保维度匹配
        if conv_out.shape[1] != sobel_out.shape[1]:
            # 如果维度不匹配，对sobel_out进行投影
            if not hasattr(self, 'project'):
                self.project = Conv(sobel_out.shape[1], conv_out.shape[1], 1, 1)
            sobel_out = self.project(sobel_out)
        
        # 拼接融合
        fused = self.fusion(torch.cat([conv_out, sobel_out], dim=1))
        
        if self.add:
            fused = x + fused
        return fused

class MambaBottleneck(nn.Module):
    def __init__(self, dim, r=1):
        super().__init__()

        self.dwt = DWTForward(wave='haar')

        # ---------- Residual branch (depthwise stride=2) ----------
        self.residual = nn.Sequential(
            nn.Conv2d(dim, dim, 3, stride=2, padding=1, groups=dim, bias=False),
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU()
        )

        # ---------- Low-frequency branch (depthwise 3x3) ----------
        self.low_dw = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU()
        )

        # ---------- High-frequency reduce ----------
        self.high_reduce = nn.Sequential(
            nn.Conv2d(3 * dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU()
        )

        # ---------- Extra downsample before Mamba (critical) ----------
        # self.mamba_down = nn.AvgPool2d(2)

        self.norm = nn.LayerNorm(dim)

        # ---------- Lightweight Mamba ----------
        self.mambas = nn.ModuleList( Mamba(
            d_model=dim ,   # 降维
            d_state=4,          # 更小
            d_conv=3,
            expand=1
        ) for _ in range(r) )
       

        # ---------- Fusion ----------
        self.fusion = nn.Sequential(
            nn.Conv2d(dim + dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU()
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # Residual
        residual = self.residual(x)

        # DWT
        # with :  # DWT不支持半精度
        LL, yh = self.dwt(x)
        LH, HL, HH = yh[0][:, :, 0], yh[0][:, :, 1], yh[0][:, :, 2]

        # Low-frequency
        low = self.low_dw(LL)

        # High-frequency
        high = torch.cat([LH, HL, HH], dim=1)
        high = self.high_reduce(high)

        # ↓↓↓ 关键降低 FLOPs ↓↓↓
        # high = self.mamba_down(high)  # H/4, W/4

        Bh, Ch, Hh, Wh = high.shape

        high = high.flatten(2).transpose(1, 2)  # B, L, C
        high = self.norm(high)

        if high.device.type == 'cuda':
            for mb in self.mambas:
                high = mb(high)

        high = high.transpose(1, 2).reshape(Bh, Ch, Hh, Wh)

        # 上采样回 LL 尺寸
        # high = nn.functional.interpolate(high, size=low.shape[-2:], mode='bilinear', align_corners=False)

        out = self.fusion(torch.cat([low, high], dim=1))

        return out + residual
class FSSB(nn.Module):
    """FSSB: Fast Spatial Squeeze-and-Excitation Block for YOLOv10 by Glenn Jocher."""

    def __init__(self, c1, c2, r=1, k=3, s=1):
        """Initializes FSSB module with given input/output channels, kernel size, and stride."""
        super().__init__()
        self.conv = DWConv(c1, c2, k, s)
        self.mamba = MambaBottleneck(c2, r)


    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        x = self.conv(x)
        # print(x.shape)
        out = self.mamba(x)
        # print(out.shape)
        return out
class FreqASFF(nn.Module):
    def __init__(self, level, in_channels_list, out_channels=None, mid_channels=64):
        super().__init__()
        self.level = level
        self.out_channels = out_channels if out_channels is not None else mid_channels
        self.weight_conv1 = nn.Conv2d(in_channels_list[0], 1, kernel_size=1)
        self.weight_conv2 = nn.Conv2d(in_channels_list[1], 1, kernel_size=1)
        # self.weight_conv3 = nn.Conv2d(in_channels_list[2], 1, kernel_size=1)
        self.reduce_layers = nn.ModuleList()
        self.up = UPFusion(in_channels_list[1], in_channels_list[0], in_channels_list[1])
    
    def forward(self, x):
        x1, x2, x_low =  x
        w1 = self.weight_conv1(x1)
        w2 = self.weight_conv2(x2)
        weights = torch.cat([w1, w2], dim=1)
        weights = F.softmax(weights, dim=1)
        w1, w2 = weights[:, 0:1], weights[:, 1:2]
        x2_up = self.up((x2, x_low))
        out = w1 * x1 + w2 * x2_up
        # print(out.shape)
        return out
        
class ASFF(nn.Module):
    def __init__(self, level, in_channels_list, out_channels=None, mid_channels=64):
        """
        Args:
            level: 目标层级 1/2/3
            in_channels_list: 三个输入通道数 [c1,c2,c3]
            mid_channels: 降维后的中间通道数（默认128），可进一步调小
            out_channels: 输出通道数，若为None则保持mid_channels
        """
        super().__init__()
        self.level = level
        self.mid_channels = min(in_channels_list)
        self.out_channels = out_channels if out_channels is not None else mid_channels

        # 降维层：分别将三个输入通道降到mid_channels
        self.reduce_layers = nn.ModuleList()
        for in_c in in_channels_list:
            if in_c != mid_channels:
                self.reduce_layers.append(DWConv(in_c, mid_channels, 1))
            else:
                self.reduce_layers.append(nn.Identity())

        # 特征对齐（尺寸调整）——此时通道已统一为mid_channels
        self.align = nn.ModuleList()
        target_hw = None  # 将根据目标level动态计算
        for i, in_c in enumerate(in_channels_list, start=1):
            align = self._make_align(i, level, mid_channels)
            self.align.append(align)

        # 权重生成：使用深度可分离卷积进一步降低计算量
        # 也可以使用普通1x1卷积，但mid_channels较小，开销已不大
        self.weight_conv1 = nn.Conv2d(mid_channels, 1, kernel_size=1)
        self.weight_conv2 = nn.Conv2d(mid_channels, 1, kernel_size=1)
        self.weight_conv3 = nn.Conv2d(mid_channels, 1, kernel_size=1)

        # 如果需要输出通道与mid_channels不同，加一个输出投影
        if self.out_channels != mid_channels:
            self.out_proj = DWConv(mid_channels, self.out_channels, 1)
        else:
            self.out_proj = nn.Identity()

    def _make_align(self, i, target_level, channels):
        diff = i - target_level
        ops = []
        if diff > 0:  # 下采样
            for _ in range(diff):
                ops.extend([
                    nn.Conv2d(channels, channels, 3, stride=2, padding=1),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True)
                ])
        elif diff < 0:  # 上采样
            scale = 2 ** (-diff)
            ops.append(nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False))
        if len(ops) == 0:
            return nn.Identity()
        elif len(ops) == 1:
            return ops[0]
        else:
            return nn.Sequential(*ops)
 
    def forward(self,x):

        x1, x2, x3 =  x
        # 降维
        x1 = self.reduce_layers[0](x1)
        x2 = self.reduce_layers[1](x2)
        x3 = self.reduce_layers[2](x3)

        # 对齐
        x1 = self.align[0](x1)
        x2 = self.align[1](x2)
        x3 = self.align[2](x3)

        # 生成权重
        w1 = self.weight_conv1(x1)
        w2 = self.weight_conv2(x2)
        w3 = self.weight_conv3(x3)
        weights = torch.cat([w1, w2, w3], dim=1)
        weights = F.softmax(weights, dim=1)
        w1, w2, w3 = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]

        # 融合
        out = w1 * x1 + w2 * x2 + w3 * x3
        out = self.out_proj(out)
        return out


class SplitFreq(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.dwt = DWTForward(wave='haar')
        self.conv_low = Conv(c1, c2, 1, 1)
        self.conv_high = Conv(3 * c1, c2, 1, 1)
    def forward(self, x):
        if isinstance(x, tuple):
            x = x[1]
        LL, yh = self.dwt(x)
        LH, HL, HH = yh[0][:, :, 0], yh[0][:, :, 1], yh[0][:, :, 2]
        low = self.conv_low(LL)
        high = self.conv_high(torch.cat([LH, HL, HH], dim=1))
        return (high, low)


# class SplitFreq(nn.Module):
#     """
#     多头频率分离模块（2D）
#     将输入通道分成 num_heads 组，每组独立进行小波变换和频带处理，最后融合。
#     输入: x (B, C, H, W)
#     输出: (B, C2, H/2, W/2)
#     """
#     def __init__(self, c1, c2, num_heads=4, wavelet='haar'):
#         super().__init__()
#         assert c1 % num_heads == 0, "c1 must be divisible by num_heads"
#         assert c2 % num_heads == 0, "c2 must be divisible by num_heads"
#         self.num_heads = num_heads
#         self.head_dim = c1 // num_heads
#         self.out_head_dim = c2 // num_heads

#         # 为每个头创建独立的小波变换和 1x1 卷积
#         self.dwt_list = nn.ModuleList([DWTForward(wave=wavelet) for _ in range(num_heads)])
#         self.conv_low_list = nn.ModuleList([nn.Conv2d(self.head_dim, self.out_head_dim, 1) for _ in range(num_heads)])
#         # 高频子带拼接后通道数为 3 * head_dim
#         self.conv_high_list = nn.ModuleList([nn.Conv2d(3 * self.head_dim, self.out_head_dim, 1) for _ in range(num_heads)])

#     def forward(self, x):
#         if isinstance(x, tuple):
#             x = x[1]

#         B, C, H, W = x.shape
#         # 按通道分组
#         xs = x.chunk(self.num_heads, dim=1)   # 每个元素 (B, head_dim, H, W)

#         low_outs = []
#         high_outs = []

#         for i in range(self.num_heads):
#             xi = xs[i]
#             # 小波变换
#             LL, yh = self.dwt_list[i](xi)
#             # yh[0] 包含三个高频子带: LH, HL, HH，每个形状 (B, head_dim, H/2, W/2)
#             LH, HL, HH = yh[0][:, :, 0], yh[0][:, :, 1], yh[0][:, :, 2]
#             high_feats = torch.cat([LH, HL, HH], dim=1)   # (B, 3*head_dim, H/2, W/2)

#             low_out = self.conv_low_list[i](LL)
#             high_out = self.conv_high_list[i](high_feats)

#             low_outs.append(low_out)
#             high_outs.append(high_out)

#         # 将所有头的低频和高频分别拼接，然后相加（也可选择拼接后过 1x1 卷积等）
#         low_total = torch.cat(low_outs, dim=1)    # (B, c2, H/2, W/2)
#         high_total = torch.cat(high_outs, dim=1)  # (B, c2, H/2, W/2)
#         out = low_total + high_total
#         return out


class UPFusion(nn.Module):
    def __init__(self, c1, c2, c3):
        super().__init__()
        # Conv()
        self.HL_conv = nn.Conv2d(c1, c2, (3,1), padding=(1,0))
        self.LH_conv = nn.Conv2d(c1, c2, (1,3), padding=(0,1))
        self.HH_conv = nn.Conv2d(c1, c2, (3,3), padding=1)
        self.bn1 = nn.BatchNorm2d(c2)
        self.bn2 = nn.BatchNorm2d(c2)
        self.bn3 = nn.BatchNorm2d(c2)
        self.idwt = DWTInverse(wave='haar')
        self.conv_fuse = Conv(c2, c3, 3, 1, 1)  
    def forward(self, x):
        high, low_x = x
        LH = F.silu(self.bn1(self.LH_conv(high)))
        HL = F.silu(self.bn2(self.HL_conv(high)))
        HH = F.silu(self.bn3(self.HH_conv(high)))
        LL = low_x[1]
        yh = [torch.stack([LH, HL, HH], dim=2)]
        # print(LH.shape, HL.shape, HH.shape)
        idwt = self.idwt((LL, yh))
        # print(idwt.shape)
        out = self.conv_fuse(idwt)
        # print(out.shape)
        return out
    
class UpFuseBlockV2(nn.Module):
    def __init__(self, c, scale=2, k_up=3, reduction=4):
        super().__init__()

        self.scale = scale
        self.k_up = k_up
        hidden = max(c // reduction, 16)

        # 1️⃣ 通道压缩（降计算）
        self.comp = nn.Conv2d(c, hidden, 1)

        # 2️⃣ kernel预测
        self.encoder = nn.Conv2d(
            hidden,
            (scale ** 2) * (k_up ** 2),
            3,
            padding=1
        )

        # 3️⃣ 通道注意力
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c // 4, 1),
            nn.ReLU(),
            nn.Conv2d(c // 4, c, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.size()

        # 通道重标定
        x = x * self.se(x)

        # kernel 预测
        kernel = self.comp(x)
        kernel = self.encoder(kernel)

        kernel = F.pixel_shuffle(kernel, self.scale)
        kernel = kernel.view(
            B,
            self.k_up * self.k_up,
            H * self.scale,
            W * self.scale
        )
        kernel = F.softmax(kernel, dim=1)

        # unfold
        x_unfold = F.unfold(
            x,
            self.k_up,
            padding=self.k_up // 2
        )

        x_unfold = x_unfold.view(
            B,
            C,
            self.k_up * self.k_up,
            H,
            W
        )

        x_unfold = x_unfold.repeat_interleave(
            self.scale,
            dim=3
        ).repeat_interleave(
            self.scale,
            dim=4
        )

        out = torch.sum(
            x_unfold * kernel.unsqueeze(1),
            dim=2
        )

        return out + F.interpolate(x, scale_factor=2, mode='bilinear')
# class UpFuseBlockV2(nn.Module):
#     def __init__(self, c_high, c_low, c_out, scale=2):
#         super().__init__()

#         # 上采样
#         self.up = CARAFEPlus(c_high, scale=scale)

#         # 通道对齐
#         self.align_high = nn.Conv2d(c_high, c_out, 1, bias=False)
#         self.align_low = nn.Conv2d(c_low, c_out, 1, bias=False)

#         # 可学习权重（BiFPN风格）
#         self.w = nn.Parameter(torch.ones(2))

#         # 融合卷积
#         self.fuse = nn.Sequential(
#             nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
#             nn.BatchNorm2d(c_out),
#             nn.SiLU()
#         )

#         # 轻量空间注意力
#         self.spatial_att = nn.Sequential(
#             nn.Conv2d(c_out, 1, 3, padding=1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):

#         x_high, x_low = x[0], x[1]
#         # 上采样
#         x_high = self.up(x_high)

#         # 通道对齐
#         x_high = self.align_high(x_high)
#         x_low = self.align_low(x_low)

#         # 权重归一化
#         w = torch.relu(self.w)
#         w = w / (w.sum() + 1e-6)

#         # 加权融合（不再concat）
#         x = w[0] * x_high + w[1] * x_low

#         # 卷积融合
#         x = self.fuse(x)

#         # 空间注意力
#         att = self.spatial_att(x)
#         x = x * att + x  # 残差式注意力
#         # print(x.shape)
#         return x
class SPDConv(nn.Module):
    """
    Space-to-Depth Downsampling Conv
    """
    def __init__(self, c1, c2, k=3):
        super().__init__()

        self.spd = nn.PixelUnshuffle(2)  # space-to-depth
        self.conv = nn.Conv2d(c1 * 4, c2, k, padding=k // 2, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.spd(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x
class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck with 2 convolutions module with arguments ch_in, ch_out, number, shortcut,
        groups, expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        if isinstance(x, tuple):
            x, _ = x
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        # print(out.shape)
        return out

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
class SplitBottleneck(nn.Module):
    """Split bottleneck."""
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Dconv(c_, c2, k[1], 1, act=g)
        # self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1, c2, s=1, e=4):
        """Initialize convolution with given parameters."""
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        """Initializes the ResNetLayer given arguments."""
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1, c2, nh=1, ec=128, gc=512, scale=False):
        """Initializes MaxSigmoidAttnBlock with specified arguments."""
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x, guide):
        """Forward process."""
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, -1, self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=512, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x, guide):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x, guide):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(self, ec=256, ch=(), ct=512, nh=8, k=3, scale=False):
        """Initializes ImagePoolingAttn with specified arguments."""
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x, text):
        """Executes attention mechanism on input tensor x and guide tensor."""
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Contrastive Head for YOLO-World compute the region-text scores according to the similarity between image and text
    features.
    """

    def __init__(self):
        """Initializes ContrastiveHead with specified region-text similarity parameters."""
        super().__init__()
        self.bias = nn.Parameter(torch.zeros([]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """
    Batch Norm Contrastive Head for YOLO-World using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        self.bias = nn.Parameter(torch.zeros([]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(nn.Module):
    """Rep bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a RepBottleneck module with customizable in/out channels, shortcut option, groups and expansion
        ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass through RepBottleneck layer."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class RepCSP(nn.Module):
    """Rep CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes RepCSP layer with given channels, repetitions, shortcut, groups and expansion ratio."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through RepCSP layer."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1, c2, c3, c4, n=1):
        """Initializes CSP-ELAN layer with specified channel sizes, repetitions, and convolutions."""
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x):
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1, c2):
        """Initializes ADown module with convolution layers to downsample input from channels c1 to c2."""
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1, c2, c3, k=5):
        """Initializes SPP-ELAN block with convolution and max pooling layers for spatial pyramid pooling."""
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x):
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class Silence(nn.Module):
    """Silence."""

    def __init__(self):
        """Initializes the Silence module."""
        super(Silence, self).__init__()

    def forward(self, x):
        """Forward pass through Silence layer."""
        return x


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):
        """Initializes the CBLinear module, passing inputs unchanged."""
        super(CBLinear, self).__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x):
        """Forward pass through CBLinear layer."""
        outs = self.conv(x).split(self.c2s, dim=1)
        return outs


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx):
        """Initializes CBFuse module with layer index for selective feature fusion."""
        super(CBFuse, self).__init__()
        self.idx = idx

    def forward(self, xs):
        """Forward pass through CBFuse layer."""
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        out = torch.sum(torch.stack(res + xs[-1:]), dim=0)
        return out


class RepVGGDW(torch.nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()
    
    def forward(self, x):
        return self.act(self.conv(x) + self.conv1(x))
    
    def forward_fuse(self, x):
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)
        
        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias
        
        conv1_w = torch.nn.functional.pad(conv1_w, [2,2,2,2])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1

class CIB(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, e=0.5, lk=False):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            Conv(2 * c_, 2 * c_, 3, g=2 * c_) if not lk else RepVGGDW(2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv1(x) if self.add else self.cv1(x)

class C2fCIB(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, lk=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))
        # print("Using CIB in C2fCIB", n)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8,
                 attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim*2 + self.head_dim, N).split([self.key_dim, self.key_dim, self.head_dim], dim=2)

        attn = (
            (q.transpose(-2, -1) @ k) * self.scale
        )
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x

class PSA(nn.Module):

    def __init__(self, c1, c2, e=0.5):
        super().__init__()
        assert(c1 == c2)
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        
        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(
            Conv(self.c, self.c*2, 1),
            Conv(self.c*2, self.c, 1, act=False)
        )
        
    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))

class SCDown(nn.Module):
    def __init__(self, c1, c2, k, s):
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x):
        return self.cv2(self.cv1(x))
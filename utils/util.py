import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.cnn import constant_init, kaiming_init
from mmcv.utils.parrots_wrapper import _BatchNorm

# l: [-50,50]
# ab: [-128, 128]
l_norm, ab_norm = 1.0, 1.0
l_mean, ab_mean = 50.0, 0

# denormalization for l
def uncenter_l(l):
    return l * l_norm + l_mean

def flow_warp(x,
              flow,
              interpolation='bilinear',
              padding_mode='zeros',
              align_corners=True):
    """Warp an image or a feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2). The last dimension is
            a two-channel, denoting the width and height relative offsets.
            Note that the values are not normalized to [-1, 1].
        interpolation (str): Interpolation mode: 'nearest' or 'bilinear'.
            Default: 'bilinear'.
        padding_mode (str): Padding mode: 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Whether align corners. Default: True.

    Returns:
        Tensor: Warped image or feature map.
    """
    if x.size()[-2:] != flow.size()[1:3]:
        raise ValueError(f'The spatial sizes of input ({x.size()[-2:]}) and '
                         f'flow ({flow.size()[1:3]}) are not the same.')
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    grid = torch.stack((grid_x, grid_y), 2).type_as(x)  # (w, h, 2)
    grid.requires_grad = False

    grid_flow = grid + flow
    # scale grid_flow to [-1,1]
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    output = F.grid_sample(
        x.float(),
        grid_flow.float(),
        mode=interpolation,
        padding_mode=padding_mode,
        align_corners=align_corners)
    return output

def make_layer(block, num_blocks, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        block (nn.module): nn.module class for basic block.
        num_blocks (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_blocks):
        layers.append(block(**kwarg))
    return nn.Sequential(*layers)


###### vgg preprocessing ######
def vgg_preprocess(tensor):
        # input is RGB tensor which ranges in [0,1]
        # output is BGR tensor which ranges in [0,255]
        tensor_bgr = torch.cat((tensor[:, 2:3, :, :], tensor[:, 1:2, :, :], tensor[:, 0:1, :, :]), dim=1)

        tensor_bgr_ml = tensor_bgr - torch.Tensor([0.40760392, 0.45795686, 0.48501961]).to(torch.device("cuda:0")).view(1, 3, 1, 1)
        
        return tensor_bgr_ml * 255.0


def default_init_weights(module, scale=1):
    """Initialize network weights.

    Args:
        modules (nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks.
    """
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            kaiming_init(m, a=0, mode='fan_in', bias=0)
            m.weight.data *= scale
        elif isinstance(m, nn.Linear):
            kaiming_init(m, a=0, mode='fan_in', bias=0)
            m.weight.data *= scale
        elif isinstance(m, _BatchNorm):
            constant_init(m.weight, val=1, bias=0)

class PixelShufflePack(nn.Module):
    """ Pixel Shuffle upsample layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        scale_factor (int): Upsample ratio.
        upsample_kernel (int): Kernel size of Conv layer to expand channels.

    Returns:
        Upsampled feature map.
    """

    def __init__(self, in_channels, out_channels, scale_factor,
                 upsample_kernel):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.upsample_kernel = upsample_kernel
        self.upsample_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels * scale_factor * scale_factor,
            self.upsample_kernel,
            padding=(self.upsample_kernel - 1) // 2)
        self.init_weights()

    def init_weights(self):
        """Initialize weights for PixelShufflePack.
        """
        default_init_weights(self, 1)

    def forward(self, x):
        """Forward function for PixelShufflePack.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        x = self.upsample_conv(x)
        x = F.pixel_shuffle(x, self.scale_factor)
        return x

def save2png(tensor_seq: torch.FloatTensor):
    print(tensor_seq.size())


##### color space
xyz_from_rgb = np.array(
    [[0.412453, 0.357580, 0.180423], [0.212671, 0.715160, 0.072169], [0.019334, 0.119193, 0.950227]]
)
rgb_from_xyz = np.array(
    [[3.24048134, -0.96925495, 0.05564664], [-1.53715152, 1.87599, -0.20404134], [-0.49853633, 0.04155593, 1.05731107]]
)


def tensor_lab2rgb(input):
    """
    n * 3* h *w
    """
    input_trans = input.transpose(1, 2).transpose(2, 3)  # n * h * w * 3
    L, a, b = input_trans[:, :, :, 0:1], input_trans[:, :, :, 1:2], input_trans[:, :, :, 2:]
    y = (L + 16.0) / 116.0
    x = (a / 500.0) + y
    z = y - (b / 200.0)

    neg_mask = z.data < 0
    z[neg_mask] = 0
    xyz = torch.cat((x, y, z), dim=3)

    mask = xyz.data > 0.2068966
    mask_xyz = xyz.clone()
    mask_xyz[mask] = torch.pow(xyz[mask], 3.0)
    mask_xyz[~mask] = (xyz[~mask] - 16.0 / 116.0) / 7.787
    mask_xyz[:, :, :, 0] = mask_xyz[:, :, :, 0] * 0.95047
    mask_xyz[:, :, :, 2] = mask_xyz[:, :, :, 2] * 1.08883

    rgb_trans = torch.mm(mask_xyz.view(-1, 3), torch.from_numpy(rgb_from_xyz).type_as(xyz)).view(
        input.size(0), input.size(2), input.size(3), 3
    )
    rgb = rgb_trans.transpose(2, 3).transpose(1, 2)

    mask = rgb > 0.0031308
    mask_rgb = rgb.clone()
    mask_rgb[mask] = 1.055 * torch.pow(rgb[mask], 1 / 2.4) - 0.055
    mask_rgb[~mask] = rgb[~mask] * 12.92

    neg_mask = mask_rgb.data < 0
    large_mask = mask_rgb.data > 1
    mask_rgb[neg_mask] = 0
    mask_rgb[large_mask] = 1
    return mask_rgb

def gray2rgb_batch(l):
    # gray image tensor to rgb image tensor
    l_uncenter = uncenter_l(l)
    l_uncenter = l_uncenter / (2 * l_mean)
    return torch.cat((l_uncenter, l_uncenter, l_uncenter), dim=1)
def normalize(tensor, mean, std):
    """Normalize a tensor image with mean and standard deviation.

    See ``Normalize`` for more details.

    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channely.

    Returns:
        Tensor: Normalized Tensor image.
    """
    if not torch.is_tensor(tensor) and tensor.ndimension() == 3:
        raise TypeError("tensor is not a torch image.")
    # TODO: make efficient
    if tensor.size(0) == 1:
        tensor.sub_(mean).div_(std)
    else:
        for t, m, s in zip(tensor, mean, std):
            t.sub_(m).div_(s)
    return tensor

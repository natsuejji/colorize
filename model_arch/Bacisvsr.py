from os import name
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import spynet
import util

class BasicVSR(nn.Module):
    def __init__(self, mid_chaneel=64, num_blocks=30):
        super(BasicVSR, self).__init__()

        self.mid_channel = mid_chaneel
        self.num_blocks = num_blocks

        #forward
        self.forwrad_prop = ResidualBlocksWithInputConv(mid_chaneel+3, mid_chaneel, num_blocks)
        #backward
        self.backwrad_prop = ResidualBlocksWithInputConv(mid_chaneel+3, mid_chaneel, num_blocks)
        # spynet
        self.flow_estimator = spynet.Spynet()
        #aggregation
        self.fusion = nn.Conv2d(
            self.mid_channel * 2, self.mid_channel, 1, 1, 0, bias=True)

        #upsampling
        self.upsample1 = util.PixelShufflePack(
            self.mid_channel, self.mid_channel, 2, upsample_kernel=3)
        self.upsample2 = util.PixelShufflePack(
            self.mid_channel, 64, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    def check_if_mirror_extended(self, lrs):
        """Check whether the input is a mirror-extended sequence.

        If mirror-extended, the i-th (i=0, ..., t-1) frame is equal to the
        (t-1-i)-th frame.

        Args:
            lrs (tensor): Input LR images with shape (n, t, c, h, w)
        """

        self.is_mirror_extended = False
        if lrs.size(1) % 2 == 0:
            lrs_1, lrs_2 = torch.chunk(lrs, 2, dim=1)
            if torch.norm(lrs_1 - lrs_2.flip(1)) == 0:
                self.is_mirror_extended = True

    def compute_flow(self, lrs):
        """Compute optical flow using SPyNet for feature warping.

        Note that if the input is an mirror-extended sequence, 'flows_forward'
        is not needed, since it is equal to 'flows_backward.flip(1)'.

        Args:
            lrs (tensor): Input LR images with shape (n, t, c, h, w)

        Return:
            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
                flows used for forward-time propagation (current to previous).
                'flows_backward' corresponds to the flows used for
                backward-time propagation (current to next).
        """

        n, t, c, h, w = lrs.size()
        #backward
        lrs_1 = lrs[:, :-1, :, :, :].reshape(-1, c, h, w)
        #forward
        lrs_2 = lrs[:, 1:, :, :, :].reshape(-1, c, h, w)
        #算光流後光流結果的數量會比原圖少一張
        flows_backward = self.flow_estimator(lrs_2, lrs_2).view(n, t - 1, 2, h, w)
        #如果是鏡像的就不重算一次 節省計算資源
        if self.is_mirror_extended:  # flows_forward = flows_backward.flip(1)
            flows_forward = None
        else:
            flows_forward = self.flow_estimator(lrs_2, lrs_1).view(n, t - 1, 2, h, w)
        #回傳前向光流和後向光流
        #n t c h w
        #0 1 2 3 5
        return flows_forward, flows_backward

    def forward(self, lrs):
        """Forward function for BasicVSR.

        Args:
            lrs (Tensor): Input LR sequence with shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """

        n, t, c, h, w = lrs.size()
        #規定輸入的HW為64
        assert h >= 64 and w >= 64, (
            'The height and width of inputs should be at least 64, '
            f'but got {h} and {w}.')
        #檢查輸入是否為鏡像的
        # check whether the input is an extended sequence
        self.check_if_mirror_extended(lrs)

        #計算前後光流
        # compute optical flow
        flows_forward, flows_backward = self.compute_flow(lrs)


        # backward-time propgation
        outputs = []
        feat_prop = lrs.new_zeros(n, self.mid_channel, h, w) # n 64 h w
        for i in range(t - 1, -1, -1):
            if i < t - 1:  # no warping required for the last timestep
                #取其中之一 自然變成四維
                flow = flows_backward[:, i, :, :, :] # n t c h w

                feat_prop = util.flow_warp(feat_prop, flow.permute(0, 2, 3, 1)) # n c h w -> n h w c
            
            feat_prop = torch.cat([lrs[:, i, :, :, :], feat_prop], dim=1)
            feat_prop = self.backwrad_prop(feat_prop)

            outputs.append(feat_prop)
        outputs = outputs[::-1]

        # forward-time propagation and upsampling
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, t):
            lr_curr = lrs[:, i, :, :, :]
            if i > 0:  # no warping required for the first timestep
                if flows_forward is not None:
                    flow = flows_forward[:, i - 1, :, :, :]
                else:
                    flow = flows_backward[:, -i, :, :, :]
                feat_prop = util.flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

            feat_prop = torch.cat([lr_curr, feat_prop], dim=1)
            feat_prop = self.forwrad_prop(feat_prop)

            # upsampling given the backward and forward features
            out = torch.cat([outputs[i], feat_prop], dim=1)
            out = self.lrelu(self.fusion(out))
            out = self.lrelu(self.upsample1(out))
            out = self.lrelu(self.upsample2(out))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            base = self.img_upsample(lr_curr)
            out += base
            outputs[i] = out

        return torch.stack(outputs, dim=1)

class ResidualBlocksWithInputConv(nn.Module):
    """Residual blocks with a convolution in front.

    Args:
        in_channels (int): Number of input channels of the first conv.
        out_channels (int): Number of channels of the residual blocks.
            Default: 64.
        num_blocks (int): Number of residual blocks. Default: 30.
    """

    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        super().__init__()

        main = []

        # a convolution used to match the channels of the residual blocks
        main.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
        main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        # residual blocks
        main.append(
            util.make_layer(
                ResidualBlockNoBN, num_blocks, mid_channels=out_channels))

        self.main = nn.Sequential(*main)

    def forward(self, feat):
        """
        Forward function for ResidualBlocksWithInputConv.

        Args:
            feat (Tensor): Input feature with shape (n, in_channels, h, w)

        Returns:
            Tensor: Output feature with shape (n, out_channels, h, w)
        """
        return self.main(feat)

class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    It has a style of:

    ::

        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        mid_channels (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Used to scale the residual before addition.
            Default: 1.0.
    """

    def __init__(self, mid_channels=64, res_scale=1.0):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)

        self.relu = nn.ReLU(inplace=True)

        # if res_scale < 1.0, use the default initialization, as in EDSR.
        # if res_scale = 1.0, use scaled kaiming_init, as in MSRResNet.
        if res_scale == 1.0:
            self.init_weights()

    def init_weights(self):
        """Initialize weights for ResidualBlockNoBN.

        Initialization methods like `kaiming_init` are for VGG-style
        modules. For modules with residual paths, using smaller std is
        better for stability and performance. We empirically use 0.1.
        See more details in "ESRGAN: Enhanced Super-Resolution Generative
        Adversarial Networks"
        """

        for m in [self.conv1, self.conv2]:
            util.default_init_weights(m, 0.1)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale
    

    

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    
    model = BasicVSR()
    #a = torch.zeros([1,7,3,64,64])
    
    print(model)

import numpy as np
import torch 
from utils.util import flow_warp
from model_arch import spynet


def make_mask(img1, img2, flow_net):
    _, t, c, h, w = img1.size()
    flow_result = flow_net(img1, img2)
    I1_raw = img1
    I12_raw = flow_warp(I1_raw, flow_result[0,:,:,:].unsqueeze(0).permute(0,2,3,1))
    mask = torch.ones((256, 448))
    b = (I12_raw[0, 0, :, :] == -1 ) & (I12_raw[0, 1, :, :] == -1) & (I12_raw[0, 2, :, :] == -1)
    mask[b] = 0
    c = abs(I12_raw - I1_raw[0, :, :, :]).sum(axis=1).squeeze(0)*255 > 50
    mask[c] = 0
    mask = np.transpose(np.expand_dims(mask.numpy(), 0), [1,2,0])



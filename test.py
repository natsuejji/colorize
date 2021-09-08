from vimeo_loader import Vimeo90k_dataset
import torch
import numpy as np
import torch.utils.data as data
from utils.util import flow_warp
import cv2
from model_arch.spynet import Spynet
from model_arch.ColorNet import ColorizationNet
from torch.utils.data.dataloader import DataLoader
from metric import PSNR
if __name__ == "__main__":
    dataset = Vimeo90k_dataset(dataset_mode='test')
    test_generator = DataLoader(dataset, shuffle=True, batch_size=1, num_workers=8)
    epoch = 100000
    model = ColorizationNet(batch_size=1)
    model.to(device=torch.device("cuda:0"))
    len_dataset = dataset.__len__()
    s_dict = torch.load(f'/aipr/colorrr/model/v7/{epoch}_color.pth')
    model.load_state_dict(s_dict['g'])
    model.eval()

    test_sample = dataset.read_img_seq('00024/0286')
    test_lr = test_sample['lr'].unsqueeze(0).to(device=torch.device("cuda:0"))
    test_ref = test_sample['ref'].unsqueeze(0).to(device=torch.device("cuda:0"))
    
    hr = model(test_lr, test_ref).squeeze(0)
    hr = hr[0,:,:,:].permute(1,2,0).detach().cpu().numpy()*255.0
    hr = cv2.cvtColor(hr, cv2.COLOR_RGB2BGR)
    cv2.imwrite('test.jpg',hr.astype(np.uint8))
    # mean_psnr = 0.0
    # print('------------開始評估模型--------------')
    # for i, sample in enumerate(test_generator):
    #     lr = sample['lr'].to(device=torch.device("cuda:0"))
    #     ref = sample['ref'].to(device=torch.device("cuda:0"))
    #     hr = sample['hr'].to(device=torch.device("cuda:0"))
    #     pred_hr, _= model(lr, ref)
    #     mean_psnr += PSNR()(hr*255.0, pred_hr*255.0).item()
    # mean_psnr /= len_dataset
    # print(f'mean psnr testing on vimeo90k-T: {mean_psnr:.6f}.')


    

    

    
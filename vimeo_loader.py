import torch
import torch.functional as F
import torch.utils.data as data
import torchvision
from torchvision import transforms
import utils.util as util
import random
import os.path as path
from PIL import Image
import skimage.color as color
import numpy as np

class Vimeo90k_dataset(data.Dataset):

    def __init__(self, 
                 root ='/aipr/vimeo_septuplet/', 
                 scale = 4,
                 degrade_mode = 'BI', 
                 dataset_mode = 'train'):
        super(Vimeo90k_dataset, self).__init__()

        if degrade_mode not in ['BI', 'BD']:
            raise ValueError(f'只有bi或bd兩種下採樣')
        if dataset_mode not in ['train', 'test']:
            raise ValueError(f'只有train或test兩種資料集')

        self.scale = scale
        self.dataset_mode = dataset_mode
        self.degrade_mode = degrade_mode
        #圖片位置
        self.root = root
        self.degrade_img_path = self.root + 'LR/' + degrade_mode + '/x4/' 
        self.img_path = self.root + 'HR/'

        self.metadata_path = self.root + 'sep_trainlist.txt' if dataset_mode == 'train' else self.root + 'sep_testlist.txt'
        self.all_clip = []
        self.num_frame = 7
        with open(self.metadata_path, 'r') as f:
            self.all_clip = [x.replace('\n', '') for x in f.readlines()]

    def preprocessing(self, gt, lr, ref):
        
        # 資料增強 flag
        hflip = random.random() < 0.5
        vflip = random.random() < 0.5 
        if hflip:
            gt = [x.transpose(Image.FLIP_LEFT_RIGHT) for x in gt]
            lr = [x.transpose(Image.FLIP_LEFT_RIGHT) for x in lr]
            ref = [x.transpose(Image.FLIP_LEFT_RIGHT) for x in ref]
        if vflip:
            gt = [x.transpose(Image.FLIP_TOP_BOTTOM) for x in gt]
            lr = [x.transpose(Image.FLIP_TOP_BOTTOM) for x in lr]
            ref = [x.transpose(Image.FLIP_TOP_BOTTOM) for x in ref]

        ref_tensor = torch.stack([transforms.ToTensor()(x) for x in ref])
        
        lr =  [transforms.Grayscale()(x) for x in lr]
        lr_tensor = [transforms.ToTensor()(x) for x in lr]
        lr_tensor = torch.stack([transforms.Lambda(lambda x : x.repeat(3,1,1))(x) for x in lr_tensor])

        gt_tensor  = torch.stack([transforms.ToTensor()(x) for x in gt])        
        return gt_tensor, lr_tensor, ref_tensor

    def normalize_lab(self ,inputs):
        inputs[0:1, :, :] = util.normalize(inputs[0:1, :, :], 50, 1)
        inputs[1:3, :, :] = util.normalize(inputs[1:3, :, :], (0, 0), (1, 1))
        return inputs

    def __getitem__(self, index):
        return self.read_img_seq(self.all_clip[index])

    def __len__(self):
        return len(self.all_clip)

    def read_img_seq(self, clip):
        hr_clip = []
        lr_ref_clip = []
        lr_source_clip = []

        for i in range(self.num_frame):
            
            cur_frame_idx = i+1
            cur_img_path = self.img_path + clip + '/im' + str(cur_frame_idx) + '.png'
            degraded_img_path = self.degrade_img_path + clip + '/im' + str(cur_frame_idx) + '.png'

            cur_img = Image.open(cur_img_path)
            degraded_img = Image.open(degraded_img_path)
            #第一幀的彩色幀當作Reference
            if i == 0 :
                lr_ref_clip.append(degraded_img)
            lr_source_clip.append(degraded_img)
            hr_clip.append(cur_img)
        
        hr_clip, lr_source_clip, lr_ref_clip = self.preprocessing(hr_clip, lr_source_clip, lr_ref_clip)

        return {    
            'hr' : hr_clip.float(),
            'lr' : lr_source_clip.float(),
            'ref' : lr_ref_clip.float().squeeze(0)
        }


if __name__ == '__main__':

    v = Vimeo90k_dataset()
    v.__getitem__(0)

 
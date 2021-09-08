import torch
from torch._C import ThroughputBenchmark
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data.dataloader import DataLoader
from loss import CharbonnierLoss
from metric import PSNR
from model_arch.ColorNet import ColorizationNet
from model_arch.GAN_models import Discriminator_x64
from prefetcher import CUDAPrefetcher
import utils.util as util
import vimeo_loader
import time 
import numpy as np 
import os
import cv2

def init_dataset(gpu_device, batch_size):
    #dataset
    test_loader = vimeo_loader.Vimeo90k_dataset(dataset_mode='test')

    train_generator = DataLoader(vimeo_loader.Vimeo90k_dataset(), shuffle=True, batch_size=batch_size, num_workers=8, pin_memory=True, drop_last=True)
    test_genetator = DataLoader(test_loader, shuffle=True, batch_size=batch_size, num_workers=8, pin_memory=True, drop_last=True)
    train_prefect = CUDAPrefetcher(train_generator)
    test_prefect = CUDAPrefetcher(test_genetator)

    return train_prefect, test_prefect, test_loader

def init_net(gpu_device):
    model = ColorizationNet()
    disc = Discriminator_x64()
    
    model.to(device=gpu_device)
    disc.to(device=gpu_device)

    model.train()
    disc.train()
    return model, disc

def init_optim(model, disc, lr_g=2e-4, lr_d= 2*1e-4):
    optimizer_g = optim.Adam([
        {'params': model.flow_estimator.parameters(), 'lr': lr_g*0.125},
        {'params': model.forwrad_prop.parameters()},
        {'params': model.backwrad_prop.parameters()},
        {'params': model.fusion.parameters()},
        {'params': model.upsample1.parameters()},
        {'params': model.upsample2.parameters()},
        {'params': model.conv_hr.parameters()},
        {'params': model.conv_last.parameters()},
        {'params': model.img_upsample.parameters()},
        {'params': model.warpNet.parameters()},
            ], lr=lr_g)
    optimizer_d = optim.Adam(filter(lambda p: p.requires_grad, disc.parameters()), lr=lr_d, betas=(0.5, 0.999))
    scheduler_g = optim.lr_scheduler.CosineAnnealingLR(optimizer_g, eta_min=1e-5, T_max=lr_g)
    scheduler_d = optim.lr_scheduler.CosineAnnealingLR(optimizer_d, eta_min=1e-7, T_max=lr_d)
    return scheduler_g, scheduler_d, optimizer_g, optimizer_d

def train(batch_size=2, num_iter=30000, test_step=5000, test_iter=100, save_iter=5000, resume=False, resume_path=None):

    if resume == True and resume_path == None :
        raise ValueError(f'require {resume_path}.')
    
    torch.backends.cudnn.benchmark = True
    scaler = torch.cuda.amp.GradScaler()

    #init net, scheduler
    gpu_device = torch.device("cuda:0")

    model, disc = init_net(gpu_device)
    train_prefect, test_prefect, vimeo = init_dataset(gpu_device, batch_size)
    scheduler_g, scheduler_d, optimizer_g, optimizer_d = init_optim(model=model, disc=disc)
    test_sample = vimeo.read_img_seq('00012/0528')
    test_lr = test_sample['lr'].unsqueeze(0).to(gpu_device)
    test_ref = test_sample['ref'].unsqueeze(0).to(gpu_device)

    #define l1 loss
    criterion = CharbonnierLoss()

    cur_epoch = 0
    if resume:
        print(f'---------load model----------')
        _state_dict = torch.load(resume_path, map_location="cuda:0")
        model.load_state_dict(_state_dict['g'])
        disc.load_state_dict(_state_dict['d'])
        scheduler_g.load_state_dict(_state_dict['schedule_g'])
        scheduler_d.load_state_dict(_state_dict['schedule_d'])
        optimizer_g.load_state_dict(_state_dict['optim_g'])
        optimizer_d.load_state_dict(_state_dict['optim_d'])
        cur_epoch = _state_dict['cur_epoch']+1

    y = 0.0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    zeros = torch.zeros([1, 3, 256, 448]).to(gpu_device)

    for iter_idx in range(cur_epoch,int(num_iter),1):
        #前五千次不訓練spynet的weight
        if iter_idx+1 == 1:
            for k,v in model.named_parameters():
                if 'flow_estimator' in k:
                    v.requires_grad = False
                if 'vgg' in k:
                    v.requires_grad = False

        elif iter_idx+1 == 5000+1:
            for k,v in model.named_parameters():
                if 'flow_estimator' in k:
                    v.requires_grad = True
                if 'vgg' in k:
                    v.requires_grad = False
                
        sample = train_prefect.next()
        hr = sample['hr']
        lr = sample['lr']
        ref = sample['ref']
        # optimizer 梯度歸零
        optimizer_g.zero_grad()
        optimizer_d.zero_grad()

        with torch.cuda.amp.autocast():

            # inference model
            pred_hr, w = model(lr, ref)

            # define loss
            disc_loss = 0
            generator_loss = 0
            l1_loss = 0
            p_loss = 0

            for i in range(batch_size):
                for t in range(7):
                    #batch t c h w 
                    cur_hr = hr[i, t, :, :, :].unsqueeze(0)
                    cur_pred_hr = pred_hr[i, t, :, :, :].unsqueeze(0)
                    if t == 0:
                        prev_hr = zeros
                        prev_pred_hr = zeros
                    else:
                        prev_hr = hr[i, t-1, :, :, :].unsqueeze(0)
                        prev_pred_hr = pred_hr[i, t-1, :, :, :].unsqueeze(0)
                    
                    # 計算 disc loss

                    y_pred_fake, feature_pred_fake = disc(torch.cat([cur_pred_hr.detach(), prev_pred_hr.detach()], dim=1))
                    y_pred_real, feature_pred_real = disc(torch.cat([cur_hr.detach(), prev_hr.detach()], dim=1))

                    if i == 0 and t == 0:
                        y = torch.ones_like(y_pred_real).to(gpu_device)

                    # disc loss
                    disc_loss = (
                        torch.mean((y_pred_real - torch.mean(y_pred_fake) - y) ** 2)
                        + torch.mean((y_pred_fake - torch.mean(y_pred_real) + y) ** 2)
                    ) / 2 
                disc_loss = disc_loss / 7.0
        scaler.scale(disc_loss).backward()
        scaler.step(optimizer_d)
        scheduler_d.step()

        # gan
        with torch.cuda.amp.autocast():
            for i in range(batch_size):
                for t in range(7):
                    #batch t c h w 
                    cur_hr = hr[i, t, :, :, :].unsqueeze(0)
                    cur_pred_hr = pred_hr[i, t, :, :, :].unsqueeze(0)
                    if t == 0:
                        prev_hr = zeros
                        prev_pred_hr = zeros
                    else:
                        prev_hr = hr[i, t-1, :, :, :].unsqueeze(0)
                        prev_pred_hr = pred_hr[i, t-1, :, :, :].unsqueeze(0)
                    
                    # 計算 gan loss
                    y_pred_fake, feature_pred_fake = disc(torch.cat([cur_pred_hr, prev_pred_hr], dim=1))
                    y_pred_real, feature_pred_real = disc(torch.cat([cur_hr.detach(), prev_hr.detach()], dim=1))
                        
                    generator_loss = (
                        (
                            torch.mean((y_pred_real - torch.mean(y_pred_fake) + y) ** 2)
                            + torch.mean((y_pred_fake - torch.mean(y_pred_real) - y) ** 2)
                        )
                        / 2
                        * 0.5
                    )

                    pred_lr_f = model.vgg.forward(cur_pred_hr, ["r52"])[0].div(255)
                    gt_f = model.vgg.forward(cur_hr, ["r52"])[0].detach().div(255)
                    
                    p_loss += torch.mean((pred_lr_f - gt_f) ** 2) * 5

            p_loss /= 7.0 * batch_size
            generator_loss /= 7.0 * batch_size
        scaler.scale(p_loss).backward(retain_graph=True) 
        scaler.scale(generator_loss).backward(retain_graph=True)        

        # 計算 l1 loss
        l1_loss = criterion(pred_hr, hr) * 20.0
        scaler.scale(l1_loss).backward()
        
        # 更新 optimizer和scheduler
        scaler.step(optimizer_g)
        scheduler_g.step()

        # 更新 scaler
        scaler.update()

        with torch.no_grad():
            # 每一百個iter印出訊息
            if (iter_idx+1) % 100  == 0:
                end.record()
                torch.cuda.synchronize()
                print(f'{iter_idx+1}/{num_iter} l1 loss : {l1_loss.item():.4f} ,gan loss : {generator_loss.item():.4f}. disc loss : {disc_loss.item():.4f}. p loss : {p_loss.item():.4f}\
                Consume time:{start.elapsed_time(end)/1000.0:.2f} second.')
                start.record()
            # 測試模型
            if iter_idx % test_step == test_step-1:
                mean_psnr = 0.0
                for i in range(batch_size):
                    for test_iter_idx in range(int(test_iter)):
                        sample = test_prefect.next()
                        hr = sample['hr'][i,:,:,:,:].unsqueeze(0)   
                        lr = sample['lr'][i,:,:,:,:].unsqueeze(0)
                        ref = sample['ref'][i,:,:,:].unsqueeze(0)
                        pred_hr,w = model(lr, ref)
                        mean_psnr = PSNR()(hr*255.0, pred_hr*255.0).item()

                #儲存圖片
                if not os.path.exists(f'img_results/color/{iter_idx+1}'):
                    os.makedirs(f'img_results/color/{iter_idx+1}')
                if not os.path.exists(f'img_results/color/wraped/{iter_idx+1}'):
                    os.makedirs(f'img_results/color/wraped/{iter_idx+1}')
                
                test_pred_hr, w = model(test_lr, test_ref)
                test_pred_hr = test_pred_hr.squeeze(0).permute([0,2,3,1])
                for i in range(7):
                    cur_frame = test_pred_hr[i,:,:,:].cpu().numpy() * 255.0
                    cur_frame = cv2.cvtColor(cur_frame, cv2.COLOR_RGB2BGR)
                    w_frame = w[i].permute([0,2,3,1]).squeeze(0).cpu().numpy() * 255.0
                    cv2.imwrite(f'img_results/color/{iter_idx+1}/{i}.jpg',cur_frame.astype(np.uint8))
                    cv2.imwrite(f'img_results/color/wraped/{iter_idx+1}/{i}.jpg',w_frame.astype(np.uint8))
                end.record()
                print(f'{iter_idx+1}/{num_iter}  psnr : {mean_psnr:.2f} \
                    Consume time:{start.elapsed_time(end)/1000.0:.2f} second.')
                start.record()
                
            if iter_idx % save_iter == save_iter-1:
                _state_dict = {
                    'g' : model.state_dict(),
                    'd' : disc.state_dict(),
                    'optim_g' : optimizer_g.state_dict(),
                    'optim_d' : optimizer_d.state_dict(),
                    'schedule_g' : scheduler_g.state_dict(),
                    'schedule_d' : scheduler_d.state_dict(),
                    'cur_epoch' : iter_idx
                }
                torch.save(_state_dict, f'./model/{iter_idx+1}_color.pth')
                        

if __name__ ==  '__main__':
    train(batch_size=2, test_iter=8, test_step= 2000, save_iter= 2000, resume=False, resume_path='./model/20000_color.pth')
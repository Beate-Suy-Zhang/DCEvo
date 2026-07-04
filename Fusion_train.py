# -*- coding: utf-8 -*-

from sleepnet import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction, DetailFeatureExtraction
from utils.dataset import H5Dataset
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
import time
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.loss import Fusionloss, cc
import kornia

import numpy as np

        
class RandomCropWithPosition(T.RandomCrop):
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, img):
        # 获取图像的宽度和高度
        width, height = img.size
        crop_height, crop_width = self.size

        # 计算裁剪的左上角的随机位置
        top = random.randint(0, height - crop_height)
        left = random.randint(0, width - crop_width)

        # 裁剪图像
        cropped_img = img.crop((left, top, left + crop_width, top + crop_height))

        # 返回裁剪后的图像和位置
        return cropped_img, top, left

class RandomCropWithInfo(T.RandomCrop):
    def __call__(self, img):
        i, j, h, w = self.get_params(img, self.size)
        img = F.crop(img, i, j, h, w)
        return img, (i, j, h, w)  # 返回图像和裁剪的位置信息

class SimpleDataSet(Dataset):
    def __init__(self, 
                 visible_path, 
                 infrared_path, 
                 phase="train", transform=None):
        self.phase = phase
        self.visible_path = visible_path
        self.infrared_path = infrared_path
        self.transform = T.Compose([RandomCropWithPosition(128),
                                   ])
        self.ttt = T.Compose([T.ToTensor()])
    def __len__(self):
        return len(self.infrared_path)

    def __getitem__(self, item):
        image_A_path = self.visible_path[item]
        image_B_path = self.infrared_path[item]
        image_A = Image.open(image_A_path).convert(mode='YCbCr')   #################################
        image_B = Image.open(image_B_path).convert(mode='RGB')
        if self.transform is not None:
            image_A, top, left = self.transform(image_A)
            image_B, _, _ = self.transform(image_B)
            image_A = self.ttt(image_A)
            image_B = self.ttt(image_B)

        name = image_A_path.replace("\\", "/").split("/")[-1].split(".")[0]

        return image_A, image_B, top, left, name
    

    @staticmethod
    def collate_fn(batch):
        images_A, images_B, top, left, name = zip(*batch)
        #print(len(images_B))
        images_A = torch.stack(images_A, dim=0)
        images_B = torch.stack(images_B, dim=0)
        return images_A, images_B, top, left, name        # position in (768, 1024)

def read_data(root: str):
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    train_root = root
    assert os.path.exists(train_root), "train root: {} does not exist.".format(train_root)

    train_images_visible_path = []
    train_images_infrared_path = []

    supported = [".jpg", ".JPG", ".png", ".PNG", ".bmp", 'tif', 'TIF']  # 支持的文件后缀类型

    train_visible_root = os.path.join(train_root, "viimages")
    train_infrared_root= os.path.join(train_root, "irimages")

    train_visible_path = [os.path.join(train_visible_root, i) for i in os.listdir(train_visible_root)
                  if os.path.splitext(i)[-1] in supported]
    train_infrared_path = [os.path.join(train_infrared_root, i) for i in os.listdir(train_infrared_root)
                  if os.path.splitext(i)[-1] in supported]

    train_visible_path.sort()
    train_infrared_path.sort()

    assert len(train_visible_path) == len(train_infrared_path),' The length of train dataset does not match. low:{}, high:{}'.\
                                         format(len(train_visible_path),len(train_infrared_path))
    print("Visible and Infrared images check finish")

    for index in range(len(train_visible_path)):
        img_visible_path=train_visible_path[index]
        img_infrared_path=train_infrared_path[index]
        train_images_visible_path.append(img_visible_path)
        train_images_infrared_path.append(img_infrared_path)

    total_dataset_nums = len(train_visible_path) + len(train_infrared_path) 
    print("{} images were found in the dataset.".format(total_dataset_nums))
    print("{} visible images for training.".format(len(train_visible_path)))
    print("{} infrared images for training.".format(len(train_infrared_path)))

    train_low_light_path_list = [train_visible_path, train_infrared_path]
    return train_low_light_path_list


def train():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    criteria_fusion = Fusionloss()
    model_str = 'CDDFuse'

    num_epochs = 120  # total epoch
    epoch_gap = 40  # epoches of Phase I

    lr = 1.5e-4
    weight_decay = 0

    batch_size = 4
    GPU_number = os.environ['CUDA_VISIBLE_DEVICES']
    coeff_mse_loss_VF = 1.  # alpha1
    coeff_mse_loss_IF = 1.
    coeff_decomp = 2.  # alpha2 and alpha4
    coeff_tv = 5.

    clip_grad_norm_value = 0.01
    optim_step = 20
    optim_gamma = 0.5

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    DIDF_Encoder = nn.DataParallel(Restormer_Encoder()).to(device)
    DIDF_Decoder = nn.DataParallel(Restormer_Decoder()).to(device)
    BaseFuseLayer = nn.DataParallel(BaseFeatureExtraction(dim=64, num_heads=8)).to(device)
    DetailFuseLayer = nn.DataParallel(DetailFeatureExtraction(num_layers=3)).to(device)

    optimizer1 = torch.optim.Adam(
        DIDF_Encoder.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer2 = torch.optim.Adam(
        DIDF_Decoder.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer3 = torch.optim.Adam(
        BaseFuseLayer.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer4 = torch.optim.Adam(
        DetailFuseLayer.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=optim_step, gamma=optim_gamma)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=optim_step, gamma=optim_gamma)
    scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=optim_step, gamma=optim_gamma)
    scheduler4 = torch.optim.lr_scheduler.StepLR(optimizer4, step_size=optim_step, gamma=optim_gamma)

    MSELoss = nn.MSELoss()
    L1Loss = nn.L1Loss()
    Loss_ssim = kornia.losses.SSIM(11, reduction='mean')


    trainloader = DataLoader(H5Dataset(r"data/MSRS_train_imgsize_128_stride_200.h5"),
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=0)

    loader = {'train': trainloader, }
    timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")

    step = 0

    torch.backends.cudnn.benchmark = True
    prev_time = time.time()

    for epoch in range(num_epochs):
        ''' train '''

        # calculate mean loss for genetic algorithm
        all_mse_loss_V = []
        all_mse_loss_I = []
        all_loss_decomp = []
        all_fusionloss = []

        for i, (data_VIS, data_IR, _, _, _) in enumerate(loader['train']):
            data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()
            data_VIS = data_VIS[:, 0:1, :, :]
            data_IR = data_IR[:, 0:1, :, :]
            DIDF_Encoder.train()
            DIDF_Decoder.train()
            BaseFuseLayer.train()
            DetailFuseLayer.train()

            DIDF_Encoder.zero_grad()
            DIDF_Decoder.zero_grad()
            BaseFuseLayer.zero_grad()
            DetailFuseLayer.zero_grad()

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()
            optimizer4.zero_grad()

            if epoch < epoch_gap:  # Phase I
                feature_V_B, feature_V_D, _ = DIDF_Encoder(data_VIS)
                feature_I_B, feature_I_D, _ = DIDF_Encoder(data_IR)
                # data_VIS_hat, _ = DIDF_Decoder(data_VIS, feature_V_B, feature_V_D)
                # data_IR_hat, _ = DIDF_Decoder(data_IR, feature_I_B, feature_I_D)
                data_VIS_hat, _ = DIDF_Decoder(([data_VIS]), feature_V_B, feature_V_D)
                data_IR_hat, _ = DIDF_Decoder(([data_IR]), feature_I_B, feature_I_D)

                cc_loss_B = cc(feature_V_B, feature_I_B)
                cc_loss_D = cc(feature_V_D, feature_I_D)
                mse_loss_V = 5 * Loss_ssim(data_VIS, data_VIS_hat) + MSELoss(data_VIS, data_VIS_hat)
                mse_loss_I = 5 * Loss_ssim(data_IR, data_IR_hat) + MSELoss(data_IR, data_IR_hat)

                # mse_loss_V = 5 * Loss_ssim(data_VIS, data_VIS_hat) #+ MSELoss(data_VIS, data_Fuse)
                # mse_loss_I = 5 * Loss_ssim(data_IR, data_IR_hat) #+ MSELoss(data_IR, data_Fuse)

                fusionloss = L1Loss(kornia.filters.SpatialGradient()(data_VIS),
                                       kornia.filters.SpatialGradient()(data_VIS_hat))

                loss_decomp = (cc_loss_D) ** 2 / (1.01 + cc_loss_B)
                
                loss = coeff_mse_loss_VF * mse_loss_V + coeff_mse_loss_IF * \
                       mse_loss_I + coeff_decomp * loss_decomp + coeff_tv * fusionloss

                loss.backward()

                all_mse_loss_V.append(mse_loss_V)
                all_mse_loss_I.append(mse_loss_I)
                all_loss_decomp.append(loss_decomp)
                all_fusionloss.append(fusionloss)
                # all_gradientloss.append(0)

                nn.utils.clip_grad_norm_(
                    DIDF_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
                nn.utils.clip_grad_norm_(
                    DIDF_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
                optimizer1.step()
                optimizer2.step()
            else:  # Phase II
                feature_V_B, feature_V_D, feature_V = DIDF_Encoder(data_VIS)
                feature_I_B, feature_I_D, feature_I = DIDF_Encoder(data_IR)
                feature_F_B = BaseFuseLayer(feature_I_B + feature_V_B)
                feature_F_D = DetailFuseLayer(feature_I_D + feature_V_D)
                data_Fuse, feature_F = DIDF_Decoder(([data_VIS, data_IR]), feature_F_B, feature_F_D)

                mse_loss_V = 2 * Loss_ssim(data_VIS, data_Fuse) #+ MSELoss(data_VIS, data_Fuse)
                mse_loss_I = 2 * Loss_ssim(data_IR, data_Fuse) #+ MSELoss(data_IR, data_Fuse)

                cc_loss_B = cc(feature_V_B, feature_I_B)
                cc_loss_D = cc(feature_V_D, feature_I_D)
                loss_decomp = (cc_loss_D) ** 2 / (1.01 + cc_loss_B)
                totalfusionloss, fusionloss, loss_grad = criteria_fusion(data_VIS, data_IR, data_Fuse)
                
                loss =  coeff_decomp * loss_decomp + totalfusionloss \
                        + coeff_mse_loss_VF * mse_loss_V + coeff_mse_loss_IF * mse_loss_I

                loss.backward()

                all_mse_loss_V.append(mse_loss_V)
                all_mse_loss_I.append(mse_loss_I)
                all_loss_decomp.append(loss_decomp)
                all_fusionloss.append(fusionloss)
                # all_gradientloss.append(loss_grad)

                nn.utils.clip_grad_norm_(
                    DIDF_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
                nn.utils.clip_grad_norm_(
                    DIDF_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
                nn.utils.clip_grad_norm_(
                    BaseFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
                nn.utils.clip_grad_norm_(
                    DetailFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
                optimizer1.step()
                optimizer2.step()
                optimizer3.step()
                optimizer4.step()


            # Determine approximate time left
            batches_done = epoch * len(loader['train']) + i
            batches_left = num_epochs * len(loader['train']) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [loss: %f] [lr: %f] ETA: %.10s"
                % (
                    epoch,
                    num_epochs,
                    i,
                    len(loader['train']),
                    loss.item(),
                    optimizer1.param_groups[0]['lr'],
                    time_left,
                )
            )

        scheduler1.step()
        scheduler2.step()
        if not epoch < epoch_gap:
            scheduler3.step()
            scheduler4.step()

        if optimizer1.param_groups[0]['lr'] <= 1e-6:
            optimizer1.param_groups[0]['lr'] = 1e-6
        if optimizer2.param_groups[0]['lr'] <= 1e-6:
            optimizer2.param_groups[0]['lr'] = 1e-6
        if optimizer3.param_groups[0]['lr'] <= 1e-6:
            optimizer3.param_groups[0]['lr'] = 1e-6
        if optimizer4.param_groups[0]['lr'] <= 1e-6:
            optimizer4.param_groups[0]['lr'] = 1e-6
        
        if True and epoch >= 100:
            checkpoint = {
                'DIDF_Encoder': DIDF_Encoder.state_dict(),
                'DIDF_Decoder': DIDF_Decoder.state_dict(),
                'BaseFuseLayer': BaseFuseLayer.state_dict(),
                'DetailFuseLayer': DetailFuseLayer.state_dict(),
            }
            torch.save(checkpoint, os.path.join("models/DCEvo_Fusion_"+str(epoch)+'.pth'))
        
        if True:
            checkpoint = {
                'DIDF_Encoder': DIDF_Encoder.state_dict(),
                'DIDF_Decoder': DIDF_Decoder.state_dict(),
                'BaseFuseLayer': BaseFuseLayer.state_dict(),
                'DetailFuseLayer': DetailFuseLayer.state_dict(),
            }
            torch.save(checkpoint, os.path.join("models/DCEvo_Fusion_regular.pth"))
        print()


if __name__ == '__main__':
    train()



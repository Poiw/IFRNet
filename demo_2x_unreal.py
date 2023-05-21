import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import numpy as np
import torch
from models.IFRNet_ours import Model
from utils import read
from imageio import mimsave
import cv2
from os.path import join as pjoin
import imageio
from tqdm import tqdm

def load_exr(path, channel = 3):

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)[..., :3]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[..., :channel]
    # if use_opencv or channel == 1:
    #     img = cv2.imread(path, cv2.IMREAD_UNCHANGED)[..., :3]
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[..., :channel]
    # else:
    #     img = imageio.imread(path, "exr")[..., :channel]

    img[np.isnan(img)] = 0
    img[np.isinf(img)] = 1000

    return img



def saveExr(path, tensor):
    tensor = tensor.detach().cpu().numpy()
    tensor = tensor.transpose(1, 2, 0)
    imageio.imwrite(path, tensor)


def loadImg(dataDir, idx, useHigh):

    if useHigh:
        img = load_exr(pjoin(dataDir, "High+SSAA", "PreTonemapHDRColor64SPP.{:04d}.exr".format(idx)))
    else:
        img = load_exr(pjoin(dataDir, "PreTonemapHDRColor.{:04d}.exr".format(idx)))

    return img


if __name__ == "__main__":

    st = 250
    en = 370
    dataDir = "/home/M2_Disk/Songyin/Data/Bunker/Train/Seq1"
    targetDir = "/mnt/SATA_DISK_1/Songyin/ExtraSS/IFRNet/Res/BunkerSeq1_Low"
    useHigh = False

    os.makedirs(targetDir, exist_ok=True)

    model = Model().cuda().eval()
    model.load_state_dict(torch.load('/mnt/SATA_DISK_1/Songyin/ExtraSS/IFRNet/Bunker/0519_2356/2023-05-20_00_37_54/latest.pth'))


    with torch.no_grad():
        for idx in tqdm(range(st, en)):

            if (idx - st) % 2 == 0:

                img = loadImg(dataDir, idx, useHigh)
                imageio.imwrite(pjoin(targetDir, "res.{:04d}.exr".format(idx-st)), img)

            else:

                img0_np = loadImg(dataDir, idx-1, useHigh)
                img1_np = loadImg(dataDir, idx+1, useHigh)
                
                img0 = (torch.tensor(img0_np.transpose(2, 0, 1)).float()).unsqueeze(0).cuda()
                img1 = (torch.tensor(img1_np.transpose(2, 0, 1)).float()).unsqueeze(0).cuda()
                
                imgt_pred = model.inference(img0, img1)

                img = imgt_pred[0].detach().cpu().numpy().transpose(1, 2, 0)
                img = cv2.resize(img, (img0_np.shape[1], img0_np.shape[0]), interpolation=cv2.INTER_CUBIC)

                
                imageio.imwrite(pjoin(targetDir, "res.{:04d}.exr".format(idx-st)), img)

                # saveExr(pjoin(targetDir, "res.{:04d}.exr".format(idx-st)), imgt_pred[0])


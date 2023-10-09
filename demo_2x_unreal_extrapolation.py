import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import numpy as np
import torch
from models.IFRNet_ours import Model_extrapolation as Model
from utils import read
from imageio import mimsave
import cv2
from os.path import join as pjoin
import imageio
from tqdm import tqdm
from utils import warp_numpy 

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
        img = load_exr(pjoin(dataDir, "High+TAA", "PreTonemapHDRColor.{:04d}.exr".format(idx)))
    else:
        img = load_exr(pjoin(dataDir, "PreTonemapHDRColor.{:04d}.exr".format(idx)))

    return img

def loadMV(dataDir, idx, useHigh):

    assert useHigh == False

    img = load_exr(pjoin(dataDir, "MotionVector.{:04d}.exr".format(idx)))[..., :2]

    return img


if __name__ == "__main__":

    # st = 1510
    # en = 1750
    # dataDir = "/mnt/SATA_DISK_1/Songyin/ExtraSS/Sequencer/Seq1"
    # targetDir = "/mnt/SATA_DISK_1/Songyin/ExtraSS/IFRNet/Res/Sequencer_Low"
    # model_path = '/mnt/SATA_DISK_1/Songyin/ExtraSS/IFRNet/Sequencer/Sequencer.pth'
    # useHigh = False

    st = 100
    en = 500
    dataDir = "/home/M2_Disk/Songyin/Data/Bunker/Train/Seq1"
    targetDir = "/mnt/SATA_DISK_1/Songyin/ExtraSS/IFRNet_Extrapolation/Test/Bunker_GT"
    model_path = '/mnt/SATA_DISK_1/Songyin/ExtraSS/IFRNet_Extrapolation/Bunker/218.pth'
    useHigh = False

    os.makedirs(targetDir, exist_ok=True)

    model = Model().cuda().eval()
    model.load_state_dict(torch.load(model_path))


    with torch.no_grad():
        for idx in tqdm(range(st, en)):

            if (idx - st) % 2 == 0:

                img = loadImg(dataDir, idx, useHigh)

                size = img.shape[:2]
                img = img[size[0]//2 - 256:size[0]//2 + 256, size[1]//2 - 256:size[1]//2 + 256, :]

                imageio.imwrite(pjoin(targetDir, "res.{:04d}.exr".format(idx-st)), img)

            else:

                img0_np = np.clip(loadImg(dataDir, idx-2, useHigh), 0, 5)
                img1_np = np.clip(loadImg(dataDir, idx-1, useHigh), 0, 5)

                flow_0 = loadMV(dataDir, idx-2, useHigh)
                flow_1 = loadMV(dataDir, idx-1, useHigh)

                flow_0[..., 0] = flow_0[..., 0] * -1
                flow_1[..., 0] = flow_1[..., 0] * -1

                size = img0_np.shape[:2]
                img0_np = img0_np[size[0]//2 - 256:size[0]//2 + 256, size[1]//2 - 256:size[1]//2 + 256, :]
                img1_np = img1_np[size[0]//2 - 256:size[0]//2 + 256, size[1]//2 - 256:size[1]//2 + 256, :]
                flow_0 = flow_0[size[0]//2 - 256:size[0]//2 + 256, size[1]//2 - 256:size[1]//2 + 256, :]
                flow_1 = flow_1[size[0]//2 - 256:size[0]//2 + 256, size[1]//2 - 256:size[1]//2 + 256, :]


                flow_0 = flow_0 + warp_numpy(flow_1, flow_0)

                flow = np.concatenate([flow_0, flow_1], axis=-1)

                
                img0 = (torch.tensor(img0_np.transpose(2, 0, 1)).float()).unsqueeze(0).cuda()
                img1 = (torch.tensor(img1_np.transpose(2, 0, 1)).float()).unsqueeze(0).cuda()
                flow = (torch.tensor(flow.transpose(2, 0, 1)).float()).unsqueeze(0).cuda()
                
                imgt_pred = model.inference(img0, img1, flow)
                imgt_pred = imgt_pred.clamp(0, 5)

                img = imgt_pred[0].detach().cpu().numpy().transpose(1, 2, 0)
                img = cv2.resize(img, (img0_np.shape[1], img0_np.shape[0]), interpolation=cv2.INTER_CUBIC)

                
                imageio.imwrite(pjoin(targetDir, "res.{:04d}.exr".format(idx-st)), img)

                # saveExr(pjoin(targetDir, "res.{:04d}.exr".format(idx-st)), imgt_pred[0])


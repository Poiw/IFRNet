import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import numpy as np
import torch
from models.IFRNet_ours import Model_Falcor_extrapolation as Model
from utils import read
from imageio import mimsave
import cv2
from os.path import join as pjoin
import imageio
from tqdm import tqdm
from utils import warp_numpy 
import config
from utils import DeToneSimple_muLaw, ToneSimple_muLaw_numpy

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



if __name__ == "__main__":

    # st = 1510
    # en = 1750
    # dataDir = "/mnt/SATA_DISK_1/Songyin/ExtraSS/Sequencer/Seq1"
    # targetDir = "/mnt/SATA_DISK_1/Songyin/ExtraSS/IFRNet/Res/Sequencer_Low"
    # model_path = '/mnt/SATA_DISK_1/Songyin/ExtraSS/IFRNet/Sequencer/Sequencer.pth'
    # useHigh = False

    st = 2613
    en = 3418
    dataDir = "/mnt/SATA_DISK_1/Songyin/RemoteRendering/Dataset/231024/BistroInterier"
    targetDir = "/mnt/SATA_DISK_1/Songyin/ExtraSS/IFRNet_Extrapolation/Test/Falcor_BistroInterier"
    model_path = '/mnt/SATA_DISK_1/Songyin/ExtraSS/IFRNet_Extrapolation/Falcor_BistroInterier/2023-10-24_18:39:04/203.pth'

    os.makedirs(targetDir, exist_ok=True)

    model = Model().cuda().eval()
    model.load_state_dict(torch.load(model_path))

    crop_size = 512

    with torch.no_grad():
        for idx in tqdm(range(st, en)):

            if (idx - st) % 2 == 1:

                img = load_exr(pjoin(dataDir, "GT.{}.exr".format(idx)))

                size = img.shape[:2]
                img = img[size[0]//2 - crop_size:size[0]//2 + crop_size, size[1]//2 - crop_size:size[1]//2 + crop_size, :]

                imageio.imwrite(pjoin(targetDir, "res.{:04d}.exr".format(idx-st)), img)

            else:

                img0_np = load_exr(pjoin(dataDir, "GT.{}.exr".format(idx-3)))
                img1_np = load_exr(pjoin(dataDir, "GT.{}.exr".format(idx-1)))

                imgt_np = load_exr(pjoin(dataDir, "Render.{}.exr".format(idx)))
                imgt_noSplat_np = load_exr(pjoin(dataDir, "Render_woSplat.{}.exr".format(idx)))

                flow_1 = load_exr(pjoin(dataDir, "MotionVector.{}.exr".format(idx)))
                flow_0 = load_exr(pjoin(dataDir, "MotionVector.{}.exr".format(idx-1)))
                flow_05 = load_exr(pjoin(dataDir, "MotionVector.{}.exr".format(idx-2)))

                flow_1[..., 0] *= flow_1.shape[1]
                flow_1[..., 1] *= flow_1.shape[0]

                flow_0[..., 0] *= flow_0.shape[1]
                flow_0[..., 1] *= flow_0.shape[0]

                flow_05[..., 0] *= flow_05.shape[1]
                flow_05[..., 1] *= flow_05.shape[0]

                flow_0 = flow_1 + warp_numpy(flow_0 + warp_numpy(flow_05, flow_0), flow_1)

                flow = np.concatenate((flow_0, flow_1), 2)   

                if config.useTonemap:
                    img0_np = ToneSimple_muLaw_numpy(img0_np * config.exposure)
                    img1_np = ToneSimple_muLaw_numpy(img1_np * config.exposure)
                    imgt_np = ToneSimple_muLaw_numpy(imgt_np * config.exposure)
                    imgt_noSplat_np = ToneSimple_muLaw_numpy(imgt_noSplat_np * config.exposure)


                img0 = np.concatenate([img0_np, imgt_np, imgt_noSplat_np], axis=-1)
                img1 = np.concatenate([img1_np, imgt_np, imgt_noSplat_np], axis=-1)

                size = img0.shape[:2]
                img0 = img0[size[0]//2 - crop_size:size[0]//2 + crop_size, size[1]//2 - crop_size:size[1]//2 + crop_size, :]
                img1 = img1[size[0]//2 - crop_size:size[0]//2 + crop_size, size[1]//2 - crop_size:size[1]//2 + crop_size, :]
                flow = flow[size[0]//2 - crop_size:size[0]//2 + crop_size, size[1]//2 - crop_size:size[1]//2 + crop_size, :]
                
                
                img0 = (torch.tensor(img0_np.transpose(2, 0, 1)).float()).unsqueeze(0).cuda()
                img1 = (torch.tensor(img1_np.transpose(2, 0, 1)).float()).unsqueeze(0).cuda()
                flow = (torch.tensor(flow.transpose(2, 0, 1)).float()).unsqueeze(0).cuda()
                
                imgt_pred, mask = model.inference(img0, img1, flow)
                imgt_pred = imgt_pred.clamp(0, 5)

                if config.useTonemap:
                    imgt_pred = DeToneSimple_muLaw(imgt_pred) / config.exposure


                img = imgt_pred[0].detach().cpu().numpy().transpose(1, 2, 0)
                mask = mask[0].detach().cpu().numpy().transpose(1, 2, 0)

                imageio.imwrite(pjoin(targetDir, "res.{:04d}.exr".format(idx-st)), img)
                imageio.imwrite(pjoin(targetDir, "mask.{:04d}.exr".format(idx-st)), mask)

                # saveExr(pjoin(targetDir, "res.{:04d}.exr".format(idx-st)), imgt_pred[0])


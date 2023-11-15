import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import numpy as np
import torch
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


    from models.IFRNet_ours import SplatNet as Model
    from datasets import Splat_Dataset

    dataset = Splat_Dataset(config.dataDir, augment=False, exposure=config.exposure, loadAll = True)

    targetDir = "/mnt/SATA_DISK_1/Songyin/RemoteRendering/Test/Bunker_SplatRefine"
    model_path = '/mnt/SATA_DISK_1/Songyin/RemoteRendering/TrainLogs/SplatNet/2023-11-14_19:23:43/latest.pth'

    os.makedirs(targetDir, exist_ok=True)

    model = Model().cuda().eval()
    model.load_state_dict(torch.load(model_path))

    with torch.no_grad():
        for idx in tqdm(range(len(dataset))):

            img, depth, gt, img_noSplat = dataset[idx]


            if idx % 2 == 1:

                if config.useTonemap:
                    gt = DeToneSimple_muLaw(gt) / config.exposure

                gt = gt.detach().cpu().numpy().transpose(1, 2, 0)

                imageio.imwrite(pjoin(targetDir, "res.{:04d}.exr".format(idx)), gt)


            else:

                img = img.unsqueeze(0).cuda()
                img_noSplat = img_noSplat.unsqueeze(0).cuda()
                depth = depth.unsqueeze(0).cuda()

                res, _, _, _ = model.inference(torch.cat([img, img_noSplat, depth], dim=1))

                if config.useTonemap:
                    res = DeToneSimple_muLaw(res) / config.exposure
                    img_noSplat = DeToneSimple_muLaw(img_noSplat) / config.exposure


                res = res[0].detach().cpu().numpy().transpose(1, 2, 0)
                img_noSplat = img_noSplat[0].detach().cpu().numpy().transpose(1, 2, 0)

                imageio.imwrite(pjoin(targetDir, "res.{:04d}.exr".format(idx)), res)
                imageio.imwrite(pjoin(targetDir, "img_noSplat.{:04d}.exr".format(idx)), img_noSplat)
                
                # imageio.imwrite(pjoin(targetDir, "residual.{:04d}.exr".format(idx-st)), residual)

                # saveExr(pjoin(targetDir, "res.{:04d}.exr".format(idx-st)), imgt_pred[0])


import os
import math
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import BGCollection_Dataset
from torch.utils.tensorboard import SummaryWriter   
from metric import calculate_psnr, calculate_ssim
from utils import AverageMeter
import logging
import imageio
import config
from utils import DeToneSimple_muLaw

def get_lr(args, iters):
    ratio = 0.5 * (1.0 + np.cos(iters / (args.epochs * args.iters_per_epoch) * math.pi))
    lr = (args.lr_start - args.lr_end) * ratio + args.lr_end
    return lr


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def saveExr(path, tensor):
    tensor = tensor.detach().cpu().numpy()
    tensor = tensor.transpose(1, 2, 0)
    imageio.imwrite(path, tensor)

def train(args, ddp_model):
    local_rank = args.local_rank
    print('Distributed Data Parallel Training IFRNet on Rank {}'.format(local_rank))

    if local_rank == 0:
        os.makedirs(args.log_path, exist_ok=True)
        log_path = os.path.join(args.log_path, time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime()))
        img_path = os.path.join(log_path, 'images')
        os.makedirs(img_path, exist_ok=True)
        os.makedirs(log_path, exist_ok=True)
        logger = logging.getLogger()
        logger.setLevel('INFO')
        BASIC_FORMAT = '%(asctime)s:%(levelname)s:%(message)s'
        DATE_FORMAT = '%Y-%m-%d_%H:%M:%S'
        formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
        chlr = logging.StreamHandler()
        chlr.setFormatter(formatter)
        chlr.setLevel('INFO')
        fhlr = logging.FileHandler(os.path.join(log_path, 'train.log'))
        fhlr.setFormatter(formatter)
        logger.addHandler(chlr)
        logger.addHandler(fhlr)
        logger.info(args)

        writer = SummaryWriter(log_dir=log_path)
        writer_iter = 0

    dataset_train = BGCollection_Dataset(config.dataDir, augment=True, exposure=config.exposure)
    sampler = DistributedSampler(dataset_train)
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=True, sampler=sampler)
    args.iters_per_epoch = dataloader_train.__len__()
    iters = args.resume_epoch * args.iters_per_epoch

    optimizer = optim.AdamW(ddp_model.parameters(), lr=args.lr_start, weight_decay=0)

    time_stamp = time.time()
    avg_rec = AverageMeter()
    avg_geo = AverageMeter()
    avg_reg = AverageMeter()

    for epoch in range(args.resume_epoch, args.epochs):
        sampler.set_epoch(epoch)
        for i, data in enumerate(dataloader_train):
            for l in range(len(data)):
                data[l] = data[l].to(args.device)
            img, depth, gt, bg, prev_img, prevprev_img = data

            img_bg = torch.cat([img, bg], dim=1)


            data_time_interval = time.time() - time_stamp
            time_stamp = time.time()

            lr = get_lr(args, iters)
            set_lr(optimizer, lr)
            optimizer.zero_grad()

            img_pred, img_warp, mask, loss_rec, loss_geo, loss_regular = ddp_model(prevprev_img, prev_img, gt, img_bg, depth)

            loss = loss_rec + loss_geo + loss_regular
            loss.backward()
            optimizer.step()

            avg_rec.update(loss_rec.cpu().data)
            avg_geo.update(loss_geo.cpu().data)
            avg_reg.update(loss_regular.cpu().data)
            train_time_interval = time.time() - time_stamp

            if (iters+1) % 10 == 0 and local_rank == 0:
                writer.add_scalar('loss', loss, writer_iter)
                writer_iter += 1

            if (iters+1) % 100 == 0 and local_rank == 0:
                logger.info('epoch:{}/{} iter:{}/{} time:{:.2f}+{:.2f} lr:{:.5e} loss_rec:{:.4e}, loss_geo:{:.4e}, loss_reg:{:.4e}'.format(epoch+1, args.epochs, iters+1, args.epochs * args.iters_per_epoch, data_time_interval, train_time_interval, lr, avg_rec.avg, avg_geo.avg, avg_reg.avg))

                if config.useTonemap:
                    img = DeToneSimple_muLaw(img) / config.exposure
                    bg = DeToneSimple_muLaw(bg) / config.exposure
                    prev_img = DeToneSimple_muLaw(prev_img) / config.exposure
                    prevprev_img = DeToneSimple_muLaw(prevprev_img) / config.exposure

                    img_pred = DeToneSimple_muLaw(img_pred) / config.exposure
                    img_warp = DeToneSimple_muLaw(img_warp) / config.exposure

                    gt = DeToneSimple_muLaw(gt) / config.exposure

                saveExr(os.path.join(img_path, '{:08d}_img.exr'.format(iters)), img[0])
                saveExr(os.path.join(img_path, '{:08d}_img_pred.exr'.format(iters)), img_pred[0])
                saveExr(os.path.join(img_path, '{:08d}_img_warp.exr'.format(iters)), img_warp[0])
                saveExr(os.path.join(img_path, '{:08d}_bg.exr'.format(iters)), bg[0])
                saveExr(os.path.join(img_path, '{:08d}_prev_img.exr'.format(iters)), prev_img[0])
                saveExr(os.path.join(img_path, '{:08d}_prevprev_img.exr'.format(iters)), prevprev_img[0])
                saveExr(os.path.join(img_path, '{:08d}_mask.exr'.format(iters)), mask[0])
                saveExr(os.path.join(img_path, '{:08d}_gt.exr'.format(iters)), gt[0])
                saveExr(os.path.join(img_path, '{:08d}_depth.exr'.format(iters)), depth[0])

                avg_rec.reset()

            iters += 1
            time_stamp = time.time()

        if (epoch+1) % args.eval_interval == 0 and local_rank == 0:
            torch.save(ddp_model.module.state_dict(), '{}/{}.pth'.format(log_path, epoch))
            torch.save(ddp_model.module.state_dict(), '{}/{}.pth'.format(log_path, 'latest'))

        dist.barrier()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IFRNet')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--eval_interval', default=1, type=int)
    parser.add_argument('--batch_size', default=6, type=int)
    parser.add_argument('--lr_start', default=1e-4, type=float)
    parser.add_argument('--lr_end', default=1e-5, type=float)
    parser.add_argument('--log_path', default='checkpoint', type=str)
    parser.add_argument('--resume_epoch', default=0, type=int)
    parser.add_argument('--resume_path', default=None, type=str)
    args = parser.parse_args()

    dist.init_process_group(backend='nccl', world_size=args.world_size)
    torch.cuda.set_device(args.local_rank)
    args.device = torch.device('cuda', args.local_rank)

    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    from models.IFRNet_ours import BGCollection_Refine_Model as Model

    args.log_path = config.log_path
    args.num_workers = args.batch_size

    model = Model().to(args.device)
    
    if args.resume_epoch != 0:
        model.load_state_dict(torch.load(args.resume_path, map_location='cpu'))
        
    ddp_model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    
    train(args, ddp_model)
    
    dist.destroy_process_group()

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import warp, get_robust_weight
from loss import *


def resize(x, scale_factor):
    return F.interpolate(x, scale_factor=scale_factor, mode="bilinear", align_corners=False)


def convrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias), 
        nn.PReLU(out_channels)
    )


class ResBlock(nn.Module):
    def __init__(self, in_channels, side_channels, bias=True):
        super(ResBlock, self).__init__()
        self.side_channels = side_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias), 
            nn.PReLU(in_channels)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(side_channels, side_channels, kernel_size=3, stride=1, padding=1, bias=bias), 
            nn.PReLU(side_channels)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias), 
            nn.PReLU(in_channels)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(side_channels, side_channels, kernel_size=3, stride=1, padding=1, bias=bias), 
            nn.PReLU(side_channels)
        )
        self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.prelu = nn.PReLU(in_channels)

    def forward(self, x):
        out = self.conv1(x)
        out[:, -self.side_channels:, :, :] = self.conv2(out[:, -self.side_channels:, :, :].clone())
        out = self.conv3(out)
        out[:, -self.side_channels:, :, :] = self.conv4(out[:, -self.side_channels:, :, :].clone())
        out = self.prelu(x + self.conv5(out))
        return out


class Encoder(nn.Module):
    def __init__(self, in_channel=3):
        super(Encoder, self).__init__()
        self.pyramid1 = nn.Sequential(
            convrelu(in_channel, 24, 3, 2, 1), 
            convrelu(24, 24, 3, 1, 1)
        )
        self.pyramid2 = nn.Sequential(
            convrelu(24, 36, 3, 2, 1), 
            convrelu(36, 36, 3, 1, 1)
        )
        self.pyramid3 = nn.Sequential(
            convrelu(36, 54, 3, 2, 1), 
            convrelu(54, 54, 3, 1, 1)
        )
        self.pyramid4 = nn.Sequential(
            convrelu(54, 72, 3, 2, 1), 
            convrelu(72, 72, 3, 1, 1)
        )
        
    def forward(self, img):
        f1 = self.pyramid1(img)
        f2 = self.pyramid2(f1)
        f3 = self.pyramid3(f2)
        f4 = self.pyramid4(f3)
        return f1, f2, f3, f4


class Decoder4(nn.Module):
    def __init__(self):
        super(Decoder4, self).__init__()
        self.convblock = nn.Sequential(
            convrelu(144, 144), 
            ResBlock(144, 24), 
            nn.ConvTranspose2d(144, 58, 4, 2, 1, bias=True)
        )
        
    def forward(self, f0, f1):
        b, c, h, w = f0.shape
        f_in = torch.cat([f0, f1], 1)
        f_out = self.convblock(f_in)
        return f_out


class Decoder4_wMV(nn.Module):
    def __init__(self):
        super(Decoder4_wMV, self).__init__()
        self.convblock = nn.Sequential(
            convrelu(148, 144), 
            ResBlock(144, 24), 
            nn.ConvTranspose2d(144, 58, 4, 2, 1, bias=True)
        )
        
    def forward(self, f0, f1, mv):
        b, c, h, w = f0.shape
        f_in = torch.cat([f0, f1, mv], 1)
        f_out = self.convblock(f_in)
        return f_out


class Decoder3(nn.Module):
    def __init__(self):
        super(Decoder3, self).__init__()
        self.convblock = nn.Sequential(
            convrelu(166, 162), 
            ResBlock(162, 24), 
            nn.ConvTranspose2d(162, 40, 4, 2, 1, bias=True)
        )

    def forward(self, ft_, f0, f1, up_flow0, up_flow1):
        f0_warp = warp(f0, up_flow0)
        f1_warp = warp(f1, up_flow1)
        f_in = torch.cat([ft_, f0_warp, f1_warp, up_flow0, up_flow1], 1)
        f_out = self.convblock(f_in)
        return f_out


class Decoder3_wMV(nn.Module):
    def __init__(self):
        super(Decoder3_wMV, self).__init__()
        self.convblock = nn.Sequential(
            convrelu(170, 162), 
            ResBlock(162, 24), 
            nn.ConvTranspose2d(162, 40, 4, 2, 1, bias=True)
        )

    def forward(self, ft_, f0, f1, up_flow0, up_flow1, mv):
        f0_warp = warp(f0, up_flow0)
        f1_warp = warp(f1, up_flow1)
        f_in = torch.cat([ft_, f0_warp, f1_warp, up_flow0, up_flow1, mv], 1)
        f_out = self.convblock(f_in)
        return f_out

class Decoder2(nn.Module):
    def __init__(self):
        super(Decoder2, self).__init__()
        self.convblock = nn.Sequential(
            convrelu(112, 108), 
            ResBlock(108, 24), 
            nn.ConvTranspose2d(108, 28, 4, 2, 1, bias=True)
        )

    def forward(self, ft_, f0, f1, up_flow0, up_flow1):
        f0_warp = warp(f0, up_flow0)
        f1_warp = warp(f1, up_flow1)
        f_in = torch.cat([ft_, f0_warp, f1_warp, up_flow0, up_flow1], 1)
        f_out = self.convblock(f_in)
        return f_out

class Decoder2_wMV(nn.Module):
    def __init__(self):
        super(Decoder2_wMV, self).__init__()
        self.convblock = nn.Sequential(
            convrelu(116, 108), 
            ResBlock(108, 24), 
            nn.ConvTranspose2d(108, 28, 4, 2, 1, bias=True)
        )

    def forward(self, ft_, f0, f1, up_flow0, up_flow1, mv):
        f0_warp = warp(f0, up_flow0)
        f1_warp = warp(f1, up_flow1)
        f_in = torch.cat([ft_, f0_warp, f1_warp, up_flow0, up_flow1, mv], 1)
        f_out = self.convblock(f_in)
        return f_out

class Decoder1(nn.Module):
    def __init__(self):
        super(Decoder1, self).__init__()
        self.convblock = nn.Sequential(
            convrelu(76, 72), 
            ResBlock(72, 24), 
            nn.ConvTranspose2d(72, 8, 4, 2, 1, bias=True)
        )
        
    def forward(self, ft_, f0, f1, up_flow0, up_flow1):
        f0_warp = warp(f0, up_flow0)
        f1_warp = warp(f1, up_flow1)
        f_in = torch.cat([ft_, f0_warp, f1_warp, up_flow0, up_flow1], 1)
        f_out = self.convblock(f_in)
        return f_out


class Decoder1_wMV(nn.Module):
    def __init__(self):
        super(Decoder1_wMV, self).__init__()
        self.convblock = nn.Sequential(
            convrelu(80, 72), 
            ResBlock(72, 24), 
            nn.ConvTranspose2d(72, 8, 4, 2, 1, bias=True)
        )
        
    def forward(self, ft_, f0, f1, up_flow0, up_flow1, mv):
        f0_warp = warp(f0, up_flow0)
        f1_warp = warp(f1, up_flow1)
        f_in = torch.cat([ft_, f0_warp, f1_warp, up_flow0, up_flow1, mv], 1)
        f_out = self.convblock(f_in)
        return f_out

class Model(nn.Module):
    def __init__(self, local_rank=-1, lr=1e-4):
        super(Model, self).__init__()
        self.encoder = Encoder()
        self.decoder4 = Decoder4()
        self.decoder3 = Decoder3()
        self.decoder2 = Decoder2()
        self.decoder1 = Decoder1()
        self.l1_loss = Charbonnier_L1()
        self.tr_loss = Ternary(7)
        self.rb_loss = Charbonnier_Ada()
        self.gc_loss = Geometry(3)


    def inference(self, img0, img1, scale_factor=1.0):
        mean_ = torch.cat([img0, img1], 2).mean(1, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        img0 = img0 - mean_
        img1 = img1 - mean_

        img0_ = resize(img0, scale_factor=scale_factor)
        img1_ = resize(img1, scale_factor=scale_factor)

        f0_1, f0_2, f0_3, f0_4 = self.encoder(img0_)
        f1_1, f1_2, f1_3, f1_4 = self.encoder(img1_)

        out4 = self.decoder4(f0_4, f1_4)
        up_flow0_4 = out4[:, 0:2]
        up_flow1_4 = out4[:, 2:4]
        ft_3_ = out4[:, 4:]

        out3 = self.decoder3(ft_3_, f0_3, f1_3, up_flow0_4, up_flow1_4)
        up_flow0_3 = out3[:, 0:2] + 2.0 * resize(up_flow0_4, scale_factor=2.0)
        up_flow1_3 = out3[:, 2:4] + 2.0 * resize(up_flow1_4, scale_factor=2.0)
        ft_2_ = out3[:, 4:]

        out2 = self.decoder2(ft_2_, f0_2, f1_2, up_flow0_3, up_flow1_3)
        up_flow0_2 = out2[:, 0:2] + 2.0 * resize(up_flow0_3, scale_factor=2.0)
        up_flow1_2 = out2[:, 2:4] + 2.0 * resize(up_flow1_3, scale_factor=2.0)
        ft_1_ = out2[:, 4:]

        out1 = self.decoder1(ft_1_, f0_1, f1_1, up_flow0_2, up_flow1_2)
        up_flow0_1 = out1[:, 0:2] + 2.0 * resize(up_flow0_2, scale_factor=2.0)
        up_flow1_1 = out1[:, 2:4] + 2.0 * resize(up_flow1_2, scale_factor=2.0)
        up_mask_1 = torch.sigmoid(out1[:, 4:5])
        up_res_1 = out1[:, 5:]

        up_flow0_1 = resize(up_flow0_1, scale_factor=(1.0/scale_factor)) * (1.0/scale_factor)
        up_flow1_1 = resize(up_flow1_1, scale_factor=(1.0/scale_factor)) * (1.0/scale_factor)
        up_mask_1 = resize(up_mask_1, scale_factor=(1.0/scale_factor))
        up_res_1 = resize(up_res_1, scale_factor=(1.0/scale_factor))

        img0_warp = warp(img0, up_flow0_1)
        img1_warp = warp(img1, up_flow1_1)
        imgt_merge = up_mask_1 * img0_warp + (1 - up_mask_1) * img1_warp + mean_
        imgt_pred = imgt_merge + up_res_1
        imgt_pred = torch.clamp(imgt_pred, 0, 1)
        return imgt_pred


    def forward(self, img0, img1, imgt, flow=None):
        mean_ = torch.cat([img0, img1], 2).mean(1, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        img0 = img0 - mean_
        img1 = img1 - mean_
        imgt_ = imgt - mean_

        f0_1, f0_2, f0_3, f0_4 = self.encoder(img0)
        f1_1, f1_2, f1_3, f1_4 = self.encoder(img1)
        ft_1, ft_2, ft_3, ft_4 = self.encoder(imgt_)

        out4 = self.decoder4(f0_4, f1_4)
        up_flow0_4 = out4[:, 0:2]
        up_flow1_4 = out4[:, 2:4]
        ft_3_ = out4[:, 4:]

        out3 = self.decoder3(ft_3_, f0_3, f1_3, up_flow0_4, up_flow1_4)
        up_flow0_3 = out3[:, 0:2] + 2.0 * resize(up_flow0_4, scale_factor=2.0)
        up_flow1_3 = out3[:, 2:4] + 2.0 * resize(up_flow1_4, scale_factor=2.0)
        ft_2_ = out3[:, 4:]

        out2 = self.decoder2(ft_2_, f0_2, f1_2, up_flow0_3, up_flow1_3)
        up_flow0_2 = out2[:, 0:2] + 2.0 * resize(up_flow0_3, scale_factor=2.0)
        up_flow1_2 = out2[:, 2:4] + 2.0 * resize(up_flow1_3, scale_factor=2.0)
        ft_1_ = out2[:, 4:]

        out1 = self.decoder1(ft_1_, f0_1, f1_1, up_flow0_2, up_flow1_2)
        up_flow0_1 = out1[:, 0:2] + 2.0 * resize(up_flow0_2, scale_factor=2.0)
        up_flow1_1 = out1[:, 2:4] + 2.0 * resize(up_flow1_2, scale_factor=2.0)
        up_mask_1 = torch.sigmoid(out1[:, 4:5])
        up_res_1 = out1[:, 5:]
        
        img0_warp = warp(img0, up_flow0_1)
        img1_warp = warp(img1, up_flow1_1)
        imgt_merge = up_mask_1 * img0_warp + (1 - up_mask_1) * img1_warp + mean_
        imgt_pred = imgt_merge + up_res_1
        imgt_pred = torch.clamp(imgt_pred, 0, 1)

        loss_rec = self.l1_loss(imgt_pred - imgt) + self.tr_loss(imgt_pred, imgt)
        loss_geo = 0.01 * (self.gc_loss(ft_1_, ft_1) + self.gc_loss(ft_2_, ft_2) + self.gc_loss(ft_3_, ft_3))

        warped_res = None
        if flow is not None:
            robust_weight0 = get_robust_weight(up_flow0_1, flow[:, 0:2], beta=0.3)
            loss_dis = 0.01 * self.rb_loss(2.0 * resize(up_flow0_2, 2.0) - flow[:, 0:2], weight=robust_weight0) 
            loss_dis += 0.01 * self.rb_loss(4.0 * resize(up_flow0_3, 4.0) - flow[:, 0:2], weight=robust_weight0) 
            loss_dis += 0.01 * self.rb_loss(8.0 * resize(up_flow0_4, 8.0) - flow[:, 0:2], weight=robust_weight0) 
            warped_res = warp(img0 + mean_, flow[:, 0:2])
        else:
            loss_dis = 0.00 * loss_geo

        return imgt_pred, warped_res, loss_rec, loss_geo, loss_dis


class Model_extrapolation(nn.Module):
    def __init__(self, local_rank=-1, lr=1e-4):
        super(Model_extrapolation, self).__init__()
        self.encoder = Encoder()
        self.decoder4 = Decoder4_wMV()
        self.decoder3 = Decoder3_wMV()
        self.decoder2 = Decoder2_wMV()
        self.decoder1 = Decoder1_wMV()
        self.l1_loss = Charbonnier_L1()
        self.tr_loss = Ternary(7)
        self.rb_loss = Charbonnier_Ada()
        self.gc_loss = Geometry(3)


    def inference(self, img0, img1, flow):
        mean_ = torch.cat([img0, img1], 2).mean(1, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        img0 = img0 - mean_
        img1 = img1 - mean_

        f0_1, f0_2, f0_3, f0_4 = self.encoder(img0)
        f1_1, f1_2, f1_3, f1_4 = self.encoder(img1)

        flow_level_1 = 0.5 * resize(flow, scale_factor=0.5)
        flow_level_2 = 0.25 * resize(flow, scale_factor=0.25)
        flow_level_3 = 0.125 * resize(flow, scale_factor=0.125)
        flow_level_4 = 0.0625 * resize(flow, scale_factor=0.0625)


        out4 = self.decoder4(f0_4, f1_4, flow_level_4)
        up_flow0_4 = out4[:, 0:2]
        up_flow1_4 = out4[:, 2:4]
        ft_3_ = out4[:, 4:]

        out3 = self.decoder3(ft_3_, f0_3, f1_3, up_flow0_4, up_flow1_4, flow_level_3)
        up_flow0_3 = out3[:, 0:2] + 2.0 * resize(up_flow0_4, scale_factor=2.0)
        up_flow1_3 = out3[:, 2:4] + 2.0 * resize(up_flow1_4, scale_factor=2.0)
        ft_2_ = out3[:, 4:]

        out2 = self.decoder2(ft_2_, f0_2, f1_2, up_flow0_3, up_flow1_3, flow_level_2)
        up_flow0_2 = out2[:, 0:2] + 2.0 * resize(up_flow0_3, scale_factor=2.0)
        up_flow1_2 = out2[:, 2:4] + 2.0 * resize(up_flow1_3, scale_factor=2.0)
        ft_1_ = out2[:, 4:]

        out1 = self.decoder1(ft_1_, f0_1, f1_1, up_flow0_2, up_flow1_2, flow_level_1)
        up_flow0_1 = out1[:, 0:2] + 2.0 * resize(up_flow0_2, scale_factor=2.0)
        up_flow1_1 = out1[:, 2:4] + 2.0 * resize(up_flow1_2, scale_factor=2.0)
        up_mask_1 = torch.sigmoid(out1[:, 4:5])
        up_res_1 = out1[:, 5:]
        
        img0_warp = warp(img0, up_flow0_1)
        img1_warp = warp(img1, up_flow1_1)
        imgt_merge = up_mask_1 * img0_warp + (1 - up_mask_1) * img1_warp + mean_
        imgt_pred = imgt_merge + up_res_1
        # imgt_pred = torch.clamp(imgt_pred, 0, 1)

        return imgt_pred


    def forward(self, img0, img1, imgt, flow):
        mean_ = torch.cat([img0, img1], 2).mean(1, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        img0 = img0 - mean_
        img1 = img1 - mean_
        imgt_ = imgt - mean_

        f0_1, f0_2, f0_3, f0_4 = self.encoder(img0)
        f1_1, f1_2, f1_3, f1_4 = self.encoder(img1)
        ft_1, ft_2, ft_3, ft_4 = self.encoder(imgt_)

        flow_level_1 = 0.5 * resize(flow, scale_factor=0.5)
        flow_level_2 = 0.25 * resize(flow, scale_factor=0.25)
        flow_level_3 = 0.125 * resize(flow, scale_factor=0.125)
        flow_level_4 = 0.0625 * resize(flow, scale_factor=0.0625)

        # # Ablation of removing flow inputs
        # flow_level_1 = flow_level_1 * 0.
        # flow_level_2 = flow_level_2 * 0.
        # flow_level_3 = flow_level_3 * 0.
        # flow_level_4 = flow_level_4 * 0.
        


        out4 = self.decoder4(f0_4, f1_4, flow_level_4)
        up_flow0_4 = out4[:, 0:2]
        up_flow1_4 = out4[:, 2:4]
        ft_3_ = out4[:, 4:]

        out3 = self.decoder3(ft_3_, f0_3, f1_3, up_flow0_4, up_flow1_4, flow_level_3)
        up_flow0_3 = out3[:, 0:2] + 2.0 * resize(up_flow0_4, scale_factor=2.0)
        up_flow1_3 = out3[:, 2:4] + 2.0 * resize(up_flow1_4, scale_factor=2.0)
        ft_2_ = out3[:, 4:]

        out2 = self.decoder2(ft_2_, f0_2, f1_2, up_flow0_3, up_flow1_3, flow_level_2)
        up_flow0_2 = out2[:, 0:2] + 2.0 * resize(up_flow0_3, scale_factor=2.0)
        up_flow1_2 = out2[:, 2:4] + 2.0 * resize(up_flow1_3, scale_factor=2.0)
        ft_1_ = out2[:, 4:]

        out1 = self.decoder1(ft_1_, f0_1, f1_1, up_flow0_2, up_flow1_2, flow_level_1)
        up_flow0_1 = out1[:, 0:2] + 2.0 * resize(up_flow0_2, scale_factor=2.0)
        up_flow1_1 = out1[:, 2:4] + 2.0 * resize(up_flow1_2, scale_factor=2.0)
        up_mask_1 = torch.sigmoid(out1[:, 4:5])
        up_res_1 = out1[:, 5:]
        
        img0_warp = warp(img0, up_flow0_1)
        img1_warp = warp(img1, up_flow1_1)
        imgt_merge = up_mask_1 * img0_warp + (1 - up_mask_1) * img1_warp + mean_
        imgt_pred = imgt_merge + up_res_1
        # imgt_pred = torch.clamp(imgt_pred, 0, 1)

        loss_rec = self.l1_loss(imgt_pred - imgt) + self.tr_loss(imgt_pred, imgt)
        loss_geo = 0.01 * (self.gc_loss(ft_1_, ft_1) + self.gc_loss(ft_2_, ft_2) + self.gc_loss(ft_3_, ft_3))
        imgt_warped = None
        if flow is not None:
            robust_weight0 = get_robust_weight(up_flow0_1, flow[:, 0:2], beta=0.3)
            robust_weight1 = get_robust_weight(up_flow1_1, flow[:, 2:4], beta=0.3)
            loss_dis = 0.01 * (self.rb_loss(2.0 * resize(up_flow0_2, 2.0) - flow[:, 0:2], weight=robust_weight0) + self.rb_loss(2.0 * resize(up_flow1_2, 2.0) - flow[:, 2:4], weight=robust_weight1))
            loss_dis += 0.01 * (self.rb_loss(4.0 * resize(up_flow0_3, 4.0) - flow[:, 0:2], weight=robust_weight0) + self.rb_loss(4.0 * resize(up_flow1_3, 4.0) - flow[:, 2:4], weight=robust_weight1))
            loss_dis += 0.01 * (self.rb_loss(8.0 * resize(up_flow0_4, 8.0) - flow[:, 0:2], weight=robust_weight0) + self.rb_loss(8.0 * resize(up_flow1_4, 8.0) - flow[:, 2:4], weight=robust_weight1))
            warped_0 = warp(img0 + mean_, flow[:, 0:2])
            warped_1 = warp(img1 + mean_, flow[:, 2:4])
            imgt_warped = torch.cat([warped_0, warped_1], 1)
        else:
            loss_dis = 0.00 * loss_geo

        return imgt_pred, imgt_warped, loss_rec, loss_geo, loss_dis
    

class Model_Falcor_extrapolation(nn.Module):
    def __init__(self, local_rank=-1, lr=1e-4):
        super(Model_Falcor_extrapolation, self).__init__()
        self.encoder = Encoder(in_channel=9)
        self.decoder4 = Decoder4_wMV()
        self.decoder3 = Decoder3_wMV()
        self.decoder2 = Decoder2_wMV()
        self.decoder1 = Decoder1_wMV()
        self.l1_loss = Charbonnier_L1()
        self.tr_loss = Ternary(7)
        self.rb_loss = Charbonnier_Ada()
        self.gc_loss = Geometry(3)


    def inference(self, img0, img1, flow):
        mean_ = torch.cat([img0, img1], 2).mean(1, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        img0 = img0 - mean_
        img1 = img1 - mean_

        f0_1, f0_2, f0_3, f0_4 = self.encoder(img0)
        f1_1, f1_2, f1_3, f1_4 = self.encoder(img1)

        flow_level_1 = 0.5 * resize(flow, scale_factor=0.5)
        flow_level_2 = 0.25 * resize(flow, scale_factor=0.25)
        flow_level_3 = 0.125 * resize(flow, scale_factor=0.125)
        flow_level_4 = 0.0625 * resize(flow, scale_factor=0.0625)


        out4 = self.decoder4(f0_4, f1_4, flow_level_4)
        up_flow0_4 = out4[:, 0:2]
        up_flow1_4 = out4[:, 2:4]
        ft_3_ = out4[:, 4:]

        out3 = self.decoder3(ft_3_, f0_3, f1_3, up_flow0_4, up_flow1_4, flow_level_3)
        up_flow0_3 = out3[:, 0:2] + 2.0 * resize(up_flow0_4, scale_factor=2.0)
        up_flow1_3 = out3[:, 2:4] + 2.0 * resize(up_flow1_4, scale_factor=2.0)
        ft_2_ = out3[:, 4:]

        out2 = self.decoder2(ft_2_, f0_2, f1_2, up_flow0_3, up_flow1_3, flow_level_2)
        up_flow0_2 = out2[:, 0:2] + 2.0 * resize(up_flow0_3, scale_factor=2.0)
        up_flow1_2 = out2[:, 2:4] + 2.0 * resize(up_flow1_3, scale_factor=2.0)
        ft_1_ = out2[:, 4:]

        out1 = self.decoder1(ft_1_, f0_1, f1_1, up_flow0_2, up_flow1_2, flow_level_1)
        up_flow0_1 = out1[:, 0:2] + 2.0 * resize(up_flow0_2, scale_factor=2.0)
        up_flow1_1 = out1[:, 2:4] + 2.0 * resize(up_flow1_2, scale_factor=2.0)
        up_mask_1 = torch.sigmoid(out1[:, 4:5])
        up_res_1 = out1[:, 5:]
        
        img0_warp = warp(img0[:, :3], up_flow0_1)
        img1_warp = warp(img1[:, :3], up_flow1_1)
        imgt_merge = up_mask_1 * img0_warp + (1 - up_mask_1) * img1_warp + mean_
        imgt_pred = imgt_merge + up_res_1
        # imgt_pred = torch.clamp(imgt_pred, 0, 1)

        return imgt_pred, up_res_1


    def forward(self, img0, img1, imgt, flow):
        mean_ = torch.cat([img0, img1], 2).mean(1, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        img0 = img0 - mean_
        img1 = img1 - mean_
        imgt_ = imgt - mean_

        f0_1, f0_2, f0_3, f0_4 = self.encoder(img0)
        f1_1, f1_2, f1_3, f1_4 = self.encoder(img1)
        ft_1, ft_2, ft_3, ft_4 = self.encoder(imgt_)

        flow_level_1 = 0.5 * resize(flow, scale_factor=0.5)
        flow_level_2 = 0.25 * resize(flow, scale_factor=0.25)
        flow_level_3 = 0.125 * resize(flow, scale_factor=0.125)
        flow_level_4 = 0.0625 * resize(flow, scale_factor=0.0625)

        # # Ablation of removing flow inputs
        # flow_level_1 = flow_level_1 * 0.
        # flow_level_2 = flow_level_2 * 0.
        # flow_level_3 = flow_level_3 * 0.
        # flow_level_4 = flow_level_4 * 0.
        


        out4 = self.decoder4(f0_4, f1_4, flow_level_4)
        up_flow0_4 = out4[:, 0:2]
        up_flow1_4 = out4[:, 2:4]
        ft_3_ = out4[:, 4:]

        out3 = self.decoder3(ft_3_, f0_3, f1_3, up_flow0_4, up_flow1_4, flow_level_3)
        up_flow0_3 = out3[:, 0:2] + 2.0 * resize(up_flow0_4, scale_factor=2.0)
        up_flow1_3 = out3[:, 2:4] + 2.0 * resize(up_flow1_4, scale_factor=2.0)
        ft_2_ = out3[:, 4:]

        out2 = self.decoder2(ft_2_, f0_2, f1_2, up_flow0_3, up_flow1_3, flow_level_2)
        up_flow0_2 = out2[:, 0:2] + 2.0 * resize(up_flow0_3, scale_factor=2.0)
        up_flow1_2 = out2[:, 2:4] + 2.0 * resize(up_flow1_3, scale_factor=2.0)
        ft_1_ = out2[:, 4:]

        out1 = self.decoder1(ft_1_, f0_1, f1_1, up_flow0_2, up_flow1_2, flow_level_1)
        up_flow0_1 = out1[:, 0:2] + 2.0 * resize(up_flow0_2, scale_factor=2.0)
        up_flow1_1 = out1[:, 2:4] + 2.0 * resize(up_flow1_2, scale_factor=2.0)
        up_mask_1 = torch.sigmoid(out1[:, 4:5])
        up_res_1 = out1[:, 5:]
        
        img0_warp = warp(img0[:, :3], up_flow0_1)
        img1_warp = warp(img1[:, :3], up_flow1_1)
        imgt_merge = up_mask_1 * img0_warp + (1 - up_mask_1) * img1_warp + mean_
        imgt_pred = imgt_merge + up_res_1
        # imgt_pred = torch.clamp(imgt_pred, 0, 1)

        loss_rec = self.l1_loss(imgt_pred - imgt[:, :3]) + self.tr_loss(imgt_pred, imgt[:, :3])
        loss_geo = 0.01 * (self.gc_loss(ft_1_, ft_1) + self.gc_loss(ft_2_, ft_2) + self.gc_loss(ft_3_, ft_3))
        imgt_warped = None
        if flow is not None:
            robust_weight0 = get_robust_weight(up_flow0_1, flow[:, 0:2], beta=0.3)
            robust_weight1 = get_robust_weight(up_flow1_1, flow[:, 2:4], beta=0.3)
            loss_dis = 0.01 * (self.rb_loss(2.0 * resize(up_flow0_2, 2.0) - flow[:, 0:2], weight=robust_weight0) + self.rb_loss(2.0 * resize(up_flow1_2, 2.0) - flow[:, 2:4], weight=robust_weight1))
            loss_dis += 0.01 * (self.rb_loss(4.0 * resize(up_flow0_3, 4.0) - flow[:, 0:2], weight=robust_weight0) + self.rb_loss(4.0 * resize(up_flow1_3, 4.0) - flow[:, 2:4], weight=robust_weight1))
            loss_dis += 0.01 * (self.rb_loss(8.0 * resize(up_flow0_4, 8.0) - flow[:, 0:2], weight=robust_weight0) + self.rb_loss(8.0 * resize(up_flow1_4, 8.0) - flow[:, 2:4], weight=robust_weight1))
            warped_0 = warp(img0[:, :3] + mean_, flow[:, 0:2])
            warped_1 = warp(img1[:, :3] + mean_, flow[:, 2:4])
            imgt_warped = torch.cat([warped_0, warped_1], 1)
        else:
            loss_dis = 0.00 * loss_geo

        return imgt_pred, imgt_warped, loss_rec, loss_geo, loss_dis
    
def KernelFilter(input, kernel):

    assert kernel.shape[1] == 9

    kernel = kernel.view(kernel.shape[0], 1, kernel.shape[1], kernel.shape[2], kernel.shape[3])

    img_shape = input.shape[2:]

    pad_input = F.pad(input, (1, 1, 1, 1), mode='reflect')

    pad_input = pad_input.view(pad_input.shape[0], pad_input.shape[1], 1, pad_input.shape[2], pad_input.shape[3])

    concated_input = torch.cat([pad_input[:, :, :, i:i+img_shape[0], j:j+img_shape[1]] for i in range(3) for j in range(3)], dim=2)

    return torch.mean(concated_input * kernel, dim=2)

class HKPNet_v2(nn.Module):

    def __init__(self, input_channel=6):
        super(HKPNet_v2, self).__init__()


        self.l1_loss = Charbonnier_L1()
        self.tr_loss = Ternary(7)

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.upconv4 = nn.Sequential(
            nn.Conv2d(256+128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.upconv3 = nn.Sequential(
            nn.Conv2d(128+96, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.upconv2 = nn.Sequential(
            nn.Conv2d(96+64, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.upconv1 = nn.Sequential(
            nn.Conv2d(64+64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.feat_conv5 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 9, kernel_size=1, stride=1, padding=0),
        )

        self.feat_conv4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 19, kernel_size=1, stride=1, padding=0),
        )

        self.feat_conv3 = nn.Sequential(
            nn.Conv2d(96, 32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 19, kernel_size=1, stride=1, padding=0),
        )

        self.feat_conv2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 19, kernel_size=1, stride=1, padding=0),
        )

        self.feat_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 19, kernel_size=1, stride=1, padding=0),
        )


        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, img0, img1, imgt, flow):

        input = img0[:, 3:]

        h, w = input.shape[2:4]
        
        if h % 16 != 0 or w % 16 != 0:

            hh = int(np.ceil(h / 16) * 16) - h
            up = hh // 2
            down = hh - up

            ww = int(np.ceil(w / 16) * 16) - w
            left = ww // 2
            right = ww - left

            input = F.pad(input, (left, right, up, down), mode='replicate')

        render = input[:, 0:3, :, :]

        x1 = self.conv1(input)
        x2 = self.conv2(self.maxpool(x1))
        x3 = self.conv3(self.maxpool(x2))
        x4 = self.conv4(self.maxpool(x3))
        x5 = self.conv5(self.maxpool(x4))

        ux4 = self.upconv4(torch.cat([self.upsample(x5), x4], dim=1))
        ux3 = self.upconv3(torch.cat([self.upsample(ux4), x3], dim=1))
        ux2 = self.upconv2(torch.cat([self.upsample(ux3), x2], dim=1))
        ux1 = self.upconv1(torch.cat([self.upsample(ux2), x1], dim=1))

        kernel5 = self.feat_conv5(x5)
        kernel4 = self.feat_conv4(ux4)
        kernel3 = self.feat_conv3(ux3)
        kernel2 = self.feat_conv2(ux2)
        kernel1 = self.feat_conv1(ux1)

        # down_k1, up_k1, blend_weight1 = torch.clamp(kernel1[:, :9, :, :], -1, 1), torch.clamp(kernel1[:, 9:18, :, :], -1, 1), torch.sigmoid(kernel1[:, 18:, :, :])
        # down_k2, up_k2, blend_weight2 = torch.clamp(kernel2[:, :9, :, :], -1, 1), torch.clamp(kernel2[:, 9:18, :, :], -1, 1), torch.sigmoid(kernel2[:, 18:, :, :])
        # down_k3, up_k3, blend_weight3 = torch.clamp(kernel3[:, :9, :, :], -1, 1), torch.clamp(kernel3[:, 9:18, :, :], -1, 1), torch.sigmoid(kernel3[:, 18:, :, :])
        # down_k4, up_k4, blend_weight4 = torch.clamp(kernel4[:, :9, :, :], -1, 1), torch.clamp(kernel4[:, 9:18, :, :], -1, 1), torch.sigmoid(kernel4[:, 18:, :, :])
        
        down_k1, up_k1, blend_weight1 = kernel1[:, :9, :, :], kernel1[:, 9:18, :, :], torch.sigmoid(kernel1[:, 18:, :, :])
        down_k2, up_k2, blend_weight2 = kernel2[:, :9, :, :], kernel2[:, 9:18, :, :], torch.sigmoid(kernel2[:, 18:, :, :])
        down_k3, up_k3, blend_weight3 = kernel3[:, :9, :, :], kernel3[:, 9:18, :, :], torch.sigmoid(kernel3[:, 18:, :, :])
        down_k4, up_k4, blend_weight4 = kernel4[:, :9, :, :], kernel4[:, 9:18, :, :], torch.sigmoid(kernel4[:, 18:, :, :])
        
        down1 = KernelFilter(render, down_k1)
        down2 = KernelFilter(self.avgpool(down1), down_k2)
        down3 = KernelFilter(self.avgpool(down2), down_k3)
        down4 = KernelFilter(self.avgpool(down3), down_k4)
        down5 = KernelFilter(self.avgpool(down4), kernel5)

        up4 = KernelFilter(self.upsample(down5), up_k4)
        up4 = up4 * blend_weight4 + down4 * (1 - blend_weight4)

        up3 = KernelFilter(self.upsample(up4), up_k3)
        up3 = up3 * blend_weight3 + down3 * (1 - blend_weight3)

        up2 = KernelFilter(self.upsample(up3), up_k2)
        up2 = up2 * blend_weight2 + down2 * (1 - blend_weight2)

        up1 = KernelFilter(self.upsample(up2), up_k1)
        up1 = up1 * blend_weight1 + down1 * (1 - blend_weight1)

        
        if w % 16 != 0 or h % 16 != 0:
            res = up1[:, :, up:up+h, left:left+w]
        else:
            res = up1

        loss_rec = self.l1_loss(res - imgt[:, :3]) + self.tr_loss(res, imgt[:, :3])
        

        return res, torch.cat([res, res], dim=1), loss_rec, torch.tensor(0.).cuda(), torch.tensor(0.).cuda()

    
    def inference(self, img0, img1, flow):

        
        input = img0[:, 3:]

        h, w = input.shape[2:4]
        
        if h % 16 != 0 or w % 16 != 0:

            hh = int(np.ceil(h / 16) * 16) - h
            up = hh // 2
            down = hh - up

            ww = int(np.ceil(w / 16) * 16) - w
            left = ww // 2
            right = ww - left

            input = F.pad(input, (left, right, up, down), mode='replicate')

        render = input[:, 0:3, :, :]

        x1 = self.conv1(input)
        x2 = self.conv2(self.maxpool(x1))
        x3 = self.conv3(self.maxpool(x2))
        x4 = self.conv4(self.maxpool(x3))
        x5 = self.conv5(self.maxpool(x4))

        ux4 = self.upconv4(torch.cat([self.upsample(x5), x4], dim=1))
        ux3 = self.upconv3(torch.cat([self.upsample(ux4), x3], dim=1))
        ux2 = self.upconv2(torch.cat([self.upsample(ux3), x2], dim=1))
        ux1 = self.upconv1(torch.cat([self.upsample(ux2), x1], dim=1))

        kernel5 = self.feat_conv5(x5)
        kernel4 = self.feat_conv4(ux4)
        kernel3 = self.feat_conv3(ux3)
        kernel2 = self.feat_conv2(ux2)
        kernel1 = self.feat_conv1(ux1)

        down_k1, up_k1, blend_weight1 = kernel1[:, :9, :, :], kernel1[:, 9:18, :, :], torch.sigmoid(kernel1[:, 18:, :, :])
        down_k2, up_k2, blend_weight2 = kernel2[:, :9, :, :], kernel2[:, 9:18, :, :], torch.sigmoid(kernel2[:, 18:, :, :])
        down_k3, up_k3, blend_weight3 = kernel3[:, :9, :, :], kernel3[:, 9:18, :, :], torch.sigmoid(kernel3[:, 18:, :, :])
        down_k4, up_k4, blend_weight4 = kernel4[:, :9, :, :], kernel4[:, 9:18, :, :], torch.sigmoid(kernel4[:, 18:, :, :])
                
        down1 = KernelFilter(render, down_k1)
        down2 = KernelFilter(self.avgpool(down1), down_k2)
        down3 = KernelFilter(self.avgpool(down2), down_k3)
        down4 = KernelFilter(self.avgpool(down3), down_k4)
        down5 = KernelFilter(self.avgpool(down4), kernel5)

        up4 = KernelFilter(self.upsample(down5), up_k4)
        up4 = up4 * blend_weight4 + down4 * (1 - blend_weight4)

        up3 = KernelFilter(self.upsample(up4), up_k3)
        up3 = up3 * blend_weight3 + down3 * (1 - blend_weight3)

        up2 = KernelFilter(self.upsample(up3), up_k2)
        up2 = up2 * blend_weight2 + down2 * (1 - blend_weight2)

        up1 = KernelFilter(self.upsample(up2), up_k1)
        up1 = up1 * blend_weight1 + down1 * (1 - blend_weight1)

        
        if w % 16 != 0 or h % 16 != 0:
            res = up1[:, :, up:up+h, left:left+w]
        else:
            res = up1

        return res


class SplatNet(nn.Module):
    def __init__(self, MaskOnlySteps = 0):
        super(SplatNet, self).__init__()

        self.layer1 = nn.Sequential(
                nn.Conv2d(in_channels  = 7,
                          out_channels = 100,
                          kernel_size = (7,7),
                          padding=(3,3)),
                nn.ReLU()
            )
            
        self.layer2 = nn.Sequential(
                nn.Conv2d(in_channels  = 100,
                          out_channels = 100,
                          kernel_size = (5,5),
                          padding=(2,2)),
                nn.ReLU()
            )

        self.layer3 = nn.Sequential(
                nn.Conv2d(in_channels  = 100,
                          out_channels = 100,
                          kernel_size = (3,3),
                          padding=(1,1)),
                nn.ReLU()
            )

        self.layer4 = nn.Sequential(
                nn.Conv2d(in_channels  = 100,
                          out_channels = 7,
                          kernel_size = (1,1)),
                nn.Sigmoid()
            )
        
        self.l1_loss = Charbonnier_L1()
        
        init_weight = nn.init.xavier_normal

        self.mask_only_steps = MaskOnlySteps

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init_weight(module.weight)

    def forward(self, inputs, gt, step):
        out1 = self.layer1(inputs)
        out = self.layer2(out1)
        out = self.layer3(out)
        out = self.layer4(out + out1)

        mask = torch.sigmoid(out[:, 0:1, :, :])
        layer1 = out[:, 1:4, :, :]
        layer2 = out[:, 4:7, :, :]

        if step < self.mask_only_steps:

            mask_gt = torch.zeros_like(mask)
            mask_gt[input[:, :1] > 0] = 1

            loss_rec = self.l1_loss(mask - mask_gt) + self.l1_loss(layer1 - input[:, :3]) + self.l1_loss(layer2 - input[:, 3:6])

            res = mask * layer1 + (1 - mask) * layer2
        else:
            res = mask * layer1 + (1 - mask) * layer2

            loss_rec = self.l1_loss(res-gt)

        return loss_rec, res, mask, layer1, layer2
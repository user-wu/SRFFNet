#!/usr/bin/python3
#coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo

def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU):
            pass
        else:
            m.initialize()


class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1      = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1        = nn.BatchNorm2d(planes)
        self.conv2      = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=(3*dilation-1)//2, bias=False, dilation=dilation)
        self.bn2        = nn.BatchNorm2d(planes)
        self.conv3      = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3        = nn.BatchNorm2d(planes*4)
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            x = self.downsample(x)
        return F.relu(out+x, inplace=True)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1    = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1      = nn.BatchNorm2d(64)
        self.layer1   = self.make_layer( 64, 3, stride=1, dilation=1)
        self.layer2   = self.make_layer(128, 4, stride=2, dilation=1)
        self.layer3   = self.make_layer(256, 6, stride=2, dilation=1)
        self.layer4   = self.make_layer(512, 3, stride=2, dilation=1)

    def make_layer(self, planes, blocks, stride, dilation):
        downsample    = nn.Sequential(nn.Conv2d(self.inplanes, planes*4, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes*4))
        layers        = [Bottleneck(self.inplanes, planes, stride, downsample, dilation=dilation)]
        self.inplanes = planes*4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out1 = F.max_pool2d(out1, kernel_size=3, stride=2, padding=1)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return out2, out3, out4, out5

    def initialize(self):
        self.load_state_dict(torch.load('../res/resnet50-19c8e357.pth'), strict=False)


class SelfModify(nn.Module):
    def __init__(self):
        super(SelfModify, self).__init__()
        self.scale = 0.1
        self.conv11 = nn.Conv2d(64, 64, kernel_size=1, stride=1)
        self.bn11 = nn.BatchNorm2d(64)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, stride=1,padding=1, dilation=1)
        self.bn12 = nn.BatchNorm2d(64)

        self.conv21 = nn.Conv2d(64, 64, kernel_size=1, stride=1)
        self.bn21 = nn.BatchNorm2d(64)
        self.conv22 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn22 = nn.BatchNorm2d(64)
        self.conv23 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=3, dilation=3)
        self.bn23 = nn.BatchNorm2d(64)

        self.conv31 = nn.Conv2d(64, 64, kernel_size=1, stride=1)
        self.bn31 = nn.BatchNorm2d(64)
        self.conv32 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn32 = nn.BatchNorm2d(64)
        self.conv33 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn33 = nn.BatchNorm2d(64)
        self.conv34 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=5, dilation=5)
        self.bn34 = nn.BatchNorm2d(64)

        self.conv41 = nn.Conv2d(64, 64, kernel_size=1, stride=1)
        self.bn41 = nn.BatchNorm2d(64)
        self.conv42 = nn.Conv2d(64, 64, kernel_size=(1, 7), stride=1)
        self.bn42 = nn.BatchNorm2d(64)
        self.conv43 = nn.Conv2d(64, 64, kernel_size=(7, 1), stride=1)
        self.bn43 = nn.BatchNorm2d(64)
        self.conv44 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=7, dilation=7)
        self.bn44 = nn.BatchNorm2d(64)

        self.convcat = nn.Conv2d(256, 64, kernel_size=1, stride=1)
        self.bncat = nn.BatchNorm2d(64)

        self.convres = nn.Conv2d(64, 64, kernel_size=1, stride=1)
        self.bnres = nn.BatchNorm2d(64)

        self.convout = nn.Conv2d(64, 64, kernel_size=1, stride=1)
        self.bnout = nn.BatchNorm2d(64)


    def forward(self, sa):
        # print(sa, '*'*100, "ORIGINAL IMAGE")
        out11 = F.relu(self.bn11(self.conv11(sa)), inplace=True)
        out12 = F.relu(self.bn12(self.conv12(out11)), inplace=True)

        # print(out12, "*"*200, 'out12')

        out21 = F.relu(self.bn21(self.conv21(sa)), inplace=True)
        out22 = F.relu(self.bn22(self.conv22(out21)), inplace=True)
        out23 = F.relu(self.bn23(self.conv23(out22)), inplace=True)

        # print(out23, "*" * 200, 'out23')

        out31 = F.relu(self.bn31(self.conv31(sa)), inplace=True)
        out32 = F.relu(self.bn32(self.conv32(out31)), inplace=True)
        out33 = F.relu(self.bn33(self.conv33(out32)), inplace=True)
        # print(out33, "*" * 200, 'out33')
        out34 = F.relu(self.bn34(self.conv34(out33)), inplace=True)

        # print(out34, "*" * 200, 'out34')

        out41 = F.relu(self.bn41(self.conv41(sa)), inplace=True)
        out42 = F.relu(self.bn42(self.conv42(out41)), inplace=True)
        out43 = F.relu(self.bn43(self.conv43(out42)), inplace=True)
        out44 = F.relu(self.bn44(self.conv44(out43)), inplace=True)

        # print(out44, "*" * 200, 'out44')

#         Concatenation + 1*1 conv

        if sa.size()[2:] != out23.size()[2:]:
            out12 = F.interpolate(out12, size=sa.size()[2:], mode='bilinear')
        if sa.size()[2:] != out23.size()[2:]:
            out23 = F.interpolate(out23, size=sa.size()[2:], mode='bilinear')
        if sa.size()[2:] != out34.size()[2:]:
            out34 = F.interpolate(out34, size=sa.size()[2:], mode='bilinear')
        if sa.size()[2:] != out44.size()[2:]:
            out44 = F.interpolate(out44, size=sa.size()[2:], mode='bilinear')


        out_cat = torch.cat((out12, out23, out34, out44),1)
        # print(out_cat, "#"*200, "outCat")
        # print(out12.size())
        # print(out23.size())
        # print(out34.size())
        # print(out44.size())
        # print(out_cat.size())

        outcat1 = F.relu(self.bncat(self.convcat(out_cat)))
        # print(outcat1.size())

#          Res Module
        outres = F.relu(self.bnres(self.convres(sa)))

        if outcat1.size()[2:] != sa.size()[2:]:
            outres = F.interpolate(outres, size=sa.size()[2:], mode='bilinear')


        # out = outcat1 + outres
        out = outcat1 + outres

        out = F.relu(self.bnout(self.convout(out)))
        # print(out)
        return out

    def initialize(self):
        weight_init(self)


class CFM(nn.Module):
    def __init__(self):
        super(CFM, self).__init__()
        self.conv1h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1h   = nn.BatchNorm2d(64)
        self.conv2h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2h   = nn.BatchNorm2d(64)
        self.conv3h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3h   = nn.BatchNorm2d(64)
        self.conv4h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4h   = nn.BatchNorm2d(64)

        self.conv1v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1v   = nn.BatchNorm2d(64)
        self.conv2v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2v   = nn.BatchNorm2d(64)
        self.conv3v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3v   = nn.BatchNorm2d(64)
        self.conv4v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4v   = nn.BatchNorm2d(64)

        self.conv1g = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1g = nn.BatchNorm2d(64)


    def forward(self, left, down, glob=None):
        if down.size()[2:] != left.size()[2:]:
            down = F.interpolate(down, size=left.size()[2:], mode='bilinear')



        out1h = F.relu(self.bn1h(self.conv1h(left )), inplace=True)
        out2h = F.relu(self.bn2h(self.conv2h(out1h)), inplace=True)
        out1v = F.relu(self.bn1v(self.conv1v(down )), inplace=True)
        out2v = F.relu(self.bn2v(self.conv2v(out1v)), inplace=True)
        fuse  = out2h * out2v

        if glob != None:
            if glob.size()[2:] != left.size()[2:]:
                glob = F.interpolate(glob, size=left.size()[2:], mode='bilinear')
            out1g = F.relu(self.bn1g(self.conv1g(glob)), inplace=True)
            out3h = F.relu(self.bn3h(self.conv3h(fuse)), inplace=True) + out1h + out1g
            out3v = F.relu(self.bn3v(self.conv3v(fuse)), inplace=True) + out1v + out1g
        else:
            out3h = F.relu(self.bn3h(self.conv3h(fuse)), inplace=True) + out1h
            out3v = F.relu(self.bn3v(self.conv3v(fuse)), inplace=True) + out1v
        out4h = F.relu(self.bn4h(self.conv4h(out3h)), inplace=True)
        out4v = F.relu(self.bn4v(self.conv4v(out3v)), inplace=True)
        return out4h, out4v

    def initialize(self):
        weight_init(self)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.cfm45  = CFM()
        self.cfm34  = CFM()
        self.cfm23  = CFM()

    def forward(self, out2h, out3h, out4h, out5v, glob, fback=None):
        if fback is not None:
            refine5      = F.interpolate(fback, size=out5v.size()[2:], mode='bilinear')
            refine4      = F.interpolate(fback, size=out4h.size()[2:], mode='bilinear')
            refine3      = F.interpolate(fback, size=out3h.size()[2:], mode='bilinear')
            refine2      = F.interpolate(fback, size=out2h.size()[2:], mode='bilinear')

            out5v = out5v+refine5
            out4h, out4v = self.cfm45(out4h+refine4, out5v, glob)
            out3h, out3v = self.cfm34(out3h+refine3, out4v, glob)
            out2h, pred  = self.cfm23(out2h+refine2, out3v, glob)
        else:
            out4h, out4v = self.cfm45(out4h, out5v, glob)
            out3h, out3v = self.cfm34(out3h, out4v, glob)
            out2h, pred  = self.cfm23(out2h, out3v, glob)
        return out2h, out3h, out4h, out5v, pred

    def initialize(self):
        weight_init(self)


class F3Net(nn.Module):
    def __init__(self, cfg):
        super(F3Net, self).__init__()
        self.cfg      = cfg
        self.bkbone   = ResNet()
        self.squeeze5 = nn.Sequential(nn.Conv2d(2048, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze4 = nn.Sequential(nn.Conv2d(1024, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze3 = nn.Sequential(nn.Conv2d( 512, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze2 = nn.Sequential(nn.Conv2d( 256, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.refiy = SelfModify()
        self.decoder1 = Decoder()
        self.decoder2 = Decoder()
        self.linearp1 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearp2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

        self.linearr2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr4 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr5 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.initialize()

    def forward(self, x, shape=None):
        out2h, out3h, out4h, out5v        = self.bkbone(x)

        out2h, out3h, out4h, out5v        = self.squeeze2(out2h), self.squeeze3(out3h), self.squeeze4(out4h), self.squeeze5(out5v)
        glob = self.refiy(out5v)
        out2h = self.refiy(out2h)
        out3h = self.refiy(out3h)
        out4h = self.refiy(out4h)
        out5v = self.refiy(out5v)
        out2h, out3h, out4h, out5v, pred1 = self.decoder1(out2h, out3h, out4h, out5v, glob)
        out2h, out3h, out4h, out5v, pred2 = self.decoder2(out2h, out3h, out4h, out5v, glob, pred1)

        shape = x.size()[2:] if shape is None else shape
        pred1 = F.interpolate(self.linearp1(pred1), size=shape, mode='bilinear')
        pred2 = F.interpolate(self.linearp2(pred2), size=shape, mode='bilinear')

        out2h = F.interpolate(self.linearr2(out2h), size=shape, mode='bilinear')
        out3h = F.interpolate(self.linearr3(out3h), size=shape, mode='bilinear')
        out4h = F.interpolate(self.linearr4(out4h), size=shape, mode='bilinear')
        out5h = F.interpolate(self.linearr5(out5v), size=shape, mode='bilinear')
        return pred1, pred2, out2h, out3h, out4h, out5h


    def initialize(self):
        if self.cfg.snapshot:
            self.load_state_dict(torch.load(self.cfg.snapshot))
        else:
            weight_init(self)

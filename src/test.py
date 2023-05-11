#!/usr/bin/python3
#coding=utf-8

import os
import sys
sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import dataset
from net import SRFFNet


class Test(object):
    def __init__(self, Dataset, Network, path):
        ## dataset
        self.cfg    = Dataset.Config(datapath=path, snapshot='./out/model-32', mode='test')
        self.data   = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=8)
        ## network
        self.net    = Network(self.cfg)
        self.net.train(False)
        self.net.cuda()

    def show(self):
        with torch.no_grad():
            for image, mask, shape, name in self.loader:
                image, mask = image.cuda().float(), mask.cuda().float()
                o2, o3, o4, o5, s2, s3, s4, s5 = self.net(image)
                plt.figure(figsize=(20, 15))
                print("Current file:", str(name)[2:-3])
                plt.subplot(3,4,1)
                plt.axis('off')
                plt.title("Image", fontsize=30)
                plt.imshow(np.uint8(image[0].permute(1,2,0).cpu().numpy()*self.cfg.std + self.cfg.mean))
                plt.subplot(3,4,2)
                plt.axis('off')
                plt.title("Ground Truth", fontsize=30)
                plt.imshow(mask[0].cpu().numpy(), cmap="gray")

                plt.subplot(3,4,5)
                plt.axis('off')
                plt.title("Res_2", fontsize=30)
                plt.imshow(o2[0, 0].cpu().numpy(), cmap="gray")

                plt.subplot(3,4,6)
                plt.axis('off')
                plt.title("Res_3", fontsize=30)
                plt.imshow(torch.sigmoid(o3[0, 0]).cpu().numpy(), cmap="gray")

                plt.subplot(3,4,7)
                plt.axis('off')
                plt.title("Res_4", fontsize=30)
                plt.imshow(o4[0, 0].cpu().numpy(), cmap="gray")

                plt.subplot(3,4,8)
                plt.axis('off')
                plt.title("Res_5", fontsize=30)
                plt.imshow(torch.sigmoid(o5[0, 0]).cpu().numpy(), cmap="gray")

                plt.subplot(349)
                plt.axis('off')
                plt.title("Res_2+SRM", fontsize=30)
                plt.imshow(s2[0, 0].cpu().numpy(), cmap="gray")

                plt.subplot(3,4,10)
                plt.axis('off')
                plt.title("Res_3+SRM", fontsize=30)
                plt.imshow(torch.sigmoid(s3[0, 0]).cpu().numpy(), cmap="gray")

                plt.subplot(3,4,11)
                plt.axis('off')
                plt.title("Res_4+SRM", fontsize=30)
                plt.imshow(s4[0, 0].cpu().numpy(), cmap="gray")

                plt.subplot(3,4,12)
                plt.axis('off')
                plt.title("Res_4+SRM", fontsize=30)
                plt.imshow(torch.sigmoid(s5[0, 0]).cpu().numpy(), cmap="gray")
                plt.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.95,
                                    wspace=0.2, hspace=0.2)
                plt.savefig(str(name)[2:-3]+'.eps', dpi=50)
                plt.show()
                input()



    def save(self):

        with torch.no_grad():
            for image, mask, shape, name in self.loader:
                image = image.cuda().float()
                out1u, out2u, out2r, out3r, out4r, out5r = self.net(image, shape)
                out   = out2u
                pred  = (torch.sigmoid(out[0,0])*255).cpu().numpy()
                head  = '../eval/maps/SRFFNet/'+ self.cfg.datapath.split('/')[-1]
                if not os.path.exists(head):
                    os.makedirs(head)
                cv2.imwrite(head+'/'+name[0]+'.png', np.round(pred))

#                 head2 = '../../PySODMetrics/data/experience_result/TEST/' + self.cfg.datapath.split('/')[-1]
#                 if not os.path.exists(head2):
#                     os.makedirs(head2)
#                 cv2.imwrite(head2 + '/' + name[0] + '.png', np.round(pred))


if __name__=='__main__':
    for path in ['../data/ECSSD', '../data/PASCAL-S', '../data/DUTS', '../data/HKU-IS', '../data/DUT-OMRON']:
    # for path in [ '../data/ECSSD']:
        t = Test(dataset, SRFFNet, path)
        t.save()
        # t.show()

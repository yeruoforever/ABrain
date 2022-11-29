import torch
import unittest 
import torchio as tio
import torch.nn as nn
import torch.optim as opt

from ...dataset.IBSR import IBSR18
from ...trainer.segmentation import SegmentationTrainer
from ...modelzoo.VNet import VNet

import os
from torch.utils.data import DataLoader

class TestSementation(unittest.TestCase):
    def setUp(self) -> None:
        trans= tio.transforms.RescaleIntensity(
            out_min_max=(-1,1),
            in_min_max=(0,180)
        )
        self.ds=IBSR18("/dev/shm/tinyset/IBSR18",transforms=trans)
        self.dl=DataLoader(self.ds,batch_size=1,num_workers=4,persistent_workers=True)
    def test_train(self):
        # for data in self.dl:
            # img = data['img'][tio.DATA]
            # seg=data['seg'][tio.DATA]
            # print(img.min(),img.max())
            # exit()
        model = VNet(4,1,32,[2,3,3,3])
        trainer = SegmentationTrainer(model,4,None,torch.device('cuda:2'))
        loss_fun = nn.CrossEntropyLoss()
        optim=opt.Adam(trainer.model.parameters(),lr=1e-4)
        for e in range(10):
            trainer.train(self.dl,loss_fun,optim,"Epoch %d"%e)

        #     print(model(img.float().to(torch.device('cuda:2'))))
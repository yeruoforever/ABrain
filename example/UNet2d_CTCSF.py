import logging
import os
from argparse import ArgumentParser
from typing import *

import nibabel as nib
import numpy as np
import pandas as pds
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from monai.losses.dice import DiceFocalLoss
from monai.networks.nets.unet import UNet
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchio import DATA, LabelMap, ScalarImage, Subject

from ABrain.dataset import CTCSF
from ABrain.trainer.watchdog import WatchDog, Sniffer


def check_and_make(d: str):
    if not os.path.exists(d):
        os.makedirs(d)


class CSFSniffer(Sniffer):
    def __init__(self) -> None:
        super().__init__()

    def sniffing(self, pred, target) -> dict:
        print(pred.shape)
        print(target.shape)
        return {"CSF": 1}

    def is_better(self, current, history) -> bool:
        return current["CSF"] > history["CSF"]


class CT2DDataset(Dataset):
    def __init__(self, ds: CTCSF, tb: pds.DataFrame) -> None:
        super().__init__()
        imgs = []
        segs = []
        for sid, st, ed in zip(tb["SubjectID"], tb["Start"], tb["End"]):
            img = ds.img_file(sid)
            seg = ds.seg_file(sid)
            img = np.asanyarray(nib.load(img).dataobj)[:, :, st:ed]
            seg = np.asanyarray(nib.load(seg).dataobj)[:, :, st:ed]
            seg[seg != 3] = 0
            seg[seg == 3] = 1
            img = torch.from_numpy(img.astype(np.int32))
            seg = torch.from_numpy(seg.astype(np.int8))
            imgs.append(img)
            segs.append(seg)
        self.imgs = torch.cat(imgs, dim=2).permute(2, 0, 1).float()
        self.segs = torch.cat(segs, dim=2).permute(2, 0, 1).long()

    def __getitem__(self, index) -> Any:
        img = self.imgs[index, None]
        seg = self.segs[index, None]
        return img, seg

    def __len__(self):
        return self.imgs.shape[0]


class ShuffleDataset(Dataset):
    def __init__(self, ds: CT2DDataset) -> None:
        super().__init__()
        self.ds = ds
        N = len(ds)
        self.ids = np.linspace(0, N - 1, N, dtype=int)
        np.random.shuffle(self.ids)

    def __getitem__(self, index) -> Any:
        index = self.ids[index]
        return self.ds[index]

    def __len__(self):
        return len(self.ids)


def gradient_backward(loss, model, scaler, optimizer):
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    nn.utils.clip_grad.clip_grad_value_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()


class Counter(object):
    def __init__(self, n_samples: int) -> None:
        self.cnt = 0
        self.N = n_samples
        self.epoch = 0

    def count(self, n):
        self.cnt += n

    def enough(self):
        return self.cnt > self.N

    def step(self):
        self.cnt -= self.N
        self.epoch += 1


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--epochs", type=int, default=300)
    argparser.add_argument("--lr", type=float, default=0.0003)
    argparser.add_argument("--batch", type=int, default=32)
    argparser.add_argument("--resume", action="store_true")
    argparser.add_argument("--gpu", type=int, default=-1)
    argparser.add_argument("--partation", action="store_true")
    argparser.add_argument("--runs", type=str, default="/tmp/yeruo/runs")
    argparser.add_argument("--fold", type=int, default=1)
    argparser.add_argument("--best-model", action="store_true")
    args = argparser.parse_args()

    dir_runs = args.runs
    dir_best = os.path.join(dir_runs, "best")
    dir_segout = os.path.join(dir_runs, "segmentation")
    dir_partation = dir_runs
    dir_cross_exp = os.path.join(dir_runs, "cross")
    file_logging = os.path.join(dir_runs, "latest_run.log")

    cross_exp = f"fold-{args.fold}"
    label_names = ["background", "CSF"]
    n_labels = len(label_names)

    check_and_make(dir_runs)
    check_and_make(dir_runs)
    check_and_make(dir_segout)

    handler = logging.FileHandler(file_logging, mode="a+")
    logging.getLogger().addHandler(handler)

    for attr in dir(args):
        if not attr.startswith("_"):
            logging.info(f"{attr}\t{getattr(args,attr)}")

    if args.gpu < 0:
        device = torch.device("cpu")
    else:
        device = torch.device(args.gpu)

    ds = CTCSF("5.0mm")

    if args.partation:
        names = []
        starts = []
        ends = []
        logging.info("Iterate over the dataset。。。")
        for sid in ds.sids:
            img = ds.img_file(sid)
            seg = ds.seg_file(sid)
            subject = Subject(name=sid, img=ScalarImage(img), seg=LabelMap(seg))
            try:
                subject.check_consistent_spatial_shape()
            except:
                logging.warning(
                    f"{sid}: spatial shape mismatch. {subject['img'].shape} vs {subject['seg'].shape}"
                )
                continue
            seg = subject["seg"][DATA]
            ids = torch.nonzero(seg == 3)[:, 3]
            st = ids.min()
            ed = ids.max()
            names.append(sid)
            starts.append(st)
            ends.append(ed)
        names = np.array(names)
        starts = np.array(starts)
        ends = np.array(ends)
        N = len(names)
        indexs = torch.linspace(0, N - 1, N, dtype=int)
        cross5 = random_split(indexs, [0.2, 0.2, 0.2, 0.2, 0.2])
        for i in range(5):
            val_set = None
            train_set = []
            for j in range(5):
                if i == j:
                    val_set = cross5[j]
                else:
                    train_set.append(cross5[j])
            train_set = [e for e in ConcatDataset(train_set)]
            val_set = [e for e in val_set]
            train_set = dict(
                SubjectID=names[train_set],
                Start=starts[train_set],
                End=ends[train_set],
            )
            val_set = dict(
                SubjectID=names[val_set], Start=starts[val_set], End=ends[val_set]
            )
            pds.DataFrame(train_set).to_csv(
                os.path.join(dir_runs, f"fold-{i+1}-train.csv")
            )
            pds.DataFrame(val_set).to_csv(
                os.path.join(dir_runs, f"fold-{i+1}-test.csv")
            )

    logging.info(f"Loading crossover experiment: {cross_exp}")
    tb_train = pds.read_csv(os.path.join(dir_partation, cross_exp + "-train.csv"))
    tb_test = pds.read_csv(os.path.join(dir_partation, cross_exp + "-test.csv"))
    logging.info(f"Loading Train Dataset...")
    ds_train = CT2DDataset(ds, tb_train)
    logging.info(f"Loading Test Dataset...")
    ds_test = CT2DDataset(ds, tb_test)

    logging.info("Loading model, optimizer, loss function...")
    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=n_labels,
        channels=[32, 64, 64, 128],
        strides=[1, 1, 1, 1],
    )
    loss_func = DiceFocalLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler()
    watchdog = WatchDog(sniffer=CSFSniffer())

    if args.resume:
        if args.best_model:
            logging.info("recover from best model.")

        else:
            logging.info("recover from latest model.")

    logging.info(f"Uploading to device {device}")
    model = model.to(device)

    logging.info(f"Traing...")
    counter = Counter(len(ds_train))
    ds_train = ConcatDataset([ShuffleDataset(ds_train) for _ in range(args.epochs)])
    loader_train = tqdm.tqdm(DataLoader(ds_train, batch_size=args.batch))
    for img, seg in loader_train:
        img = img.to(device)
        seg = seg.to(device)
        out = model(img)
        loss = loss_func(out, seg)
        gradient_backward(loss, model, scaler, optimizer)
        loader_train.set_description_str(f"loss:{loss.item()}")
        watchdog.catch(loss, out, seg)
        counter.count(seg.shape[0])
        if counter.enough():
            counter.step()
            # TODO: test
            for img, seg in DataLoader(ds_test, batch_size=args.batch):
                print(img.shape)
            logging.info(f"Epoch {counter.epoch} finished.")

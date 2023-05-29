import logging
import os
from argparse import ArgumentParser
from typing import *

import nibabel as nib
import numpy as np
import pandas as pds
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as trsf
import tqdm
from monai.losses.dice import DiceFocalLoss
from monai.networks.nets.unet import UNet
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    TensorDataset,
    random_split,
)
from torchio import DATA, LabelMap, ScalarImage, Subject

from ABrain.dataset import CTCSF
from ABrain.evaluation.functional import dice
from ABrain.trainer.watchdog import Sniffer, WatchDog


def check_and_make(d: str):
    if not os.path.exists(d):
        os.makedirs(d)


class CSFSniffer(Sniffer):
    def __init__(self, label_names: Sequence[str]) -> None:
        super().__init__()
        self.names = label_names

    def sniffing(self, pred: torch.Tensor, target) -> dict:
        with torch.no_grad():
            pred = pred.argmax(dim=1)
            target = target.squeeze(dim=1)
            res = dice(pred, target, n_labels=len(self.names))
        return {n: s for n, s in zip(self.names, res)}

    def is_better(self, current, history) -> bool:
        return current["CSF"] > history["CSF"]


class CT2DDataset(Dataset):
    def __init__(self, ds: CTCSF, tb: pds.DataFrame, use_all: bool = True) -> None:
        super().__init__()
        imgs_p = []
        segs_p = []
        imgs_n = []
        self.use_all = use_all
        for sid, st, ed in zip(tb["SubjectID"], tb["Start"], tb["End"]):
            img = ds.img_file(sid)
            seg = ds.seg_file(sid)
            img = np.asanyarray(nib.load(img).dataobj)
            seg = np.asanyarray(nib.load(seg).dataobj)
            seg[seg != 3] = 0
            seg[seg == 3] = 1
            img_p = img[:, :, st:ed]
            seg_p = seg[:, :, st:ed]
            imgs_p.append(torch.from_numpy(img_p.astype(np.int32)))
            segs_p.append(torch.from_numpy(seg_p.astype(np.int8)))
            if use_all:
                img_n = img[:, :, :st]
                imgs_n.append(torch.from_numpy(img_n.astype(np.int32)))
                img_n = img[:, :, ed:]
                imgs_n.append(torch.from_numpy(img_n.astype(np.int32)))
        self.imgs_p = torch.cat(imgs_p, dim=2).permute(2, 0, 1).float()
        self.segs_p = torch.cat(segs_p, dim=2).permute(2, 0, 1).long()
        if use_all:
            self.imgs_n = torch.cat(imgs_n, dim=2).permute(2, 0, 1).float()
            self.segs_n = torch.zeros(self.imgs_n.shape, dtype=torch.long)

    def __getitem__(self, index) -> Any:
        img = self.imgs_p[index, None]
        seg = self.segs_p[index, None]
        if self.use_all:
            index = torch.randint(self.imgs_n.shape[0], (1,)).item()
            img_n = self.imgs_n[index, None]
            seg_n = self.segs_n[index, None]
            img = torch.cat((img, img_n), dim=0)
            seg = torch.cat((seg, seg_n), dim=0)
        return img, seg

    def __len__(self):
        return self.imgs_p.shape[0]


class ShuffleDataset(Dataset):
    def __init__(self, ds: CT2DDataset, transforms: Optional[Callable] = None) -> None:
        super().__init__()
        self.ds = ds
        N = len(ds)
        self.ids = np.linspace(0, N - 1, N, dtype=int)
        np.random.shuffle(self.ids)
        self.trans = transforms

    def __getitem__(self, index) -> Any:
        index = self.ids[index]
        img, seg = self.ds[index]
        if self.trans:
            img, seg = self.trans(img, seg)
        return img, seg

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
    def __init__(self, n_samples: int, start: int = 0) -> None:
        self.cnt = 0
        self.N = n_samples
        self.epoch = start

    def count(self, n):
        self.cnt += n

    def enough(self):
        return self.cnt > self.N

    def step(self):
        self.cnt -= self.N
        self.epoch += 1


class Augment(object):
    def __init__(self, crop_size=256) -> None:
        self.p_flip = 0.3
        self.p_scale = 0.3
        self.p_rotate = 0.3
        self.p_shear = 0.3

        self.range_scale = 0.3
        self.range_rotate = 30
        self.range_shear = 10
        self.crop_size = crop_size

    def normalize(self, img: torch.Tensor, seg):
        if not isinstance(seg, torch.Tensor):
            seg = torch.from_numpy(seg)
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img).float()
        # img = torch.clamp_(img, -10, 244)
        # img = (img + 10) / 256
        return img, seg

    def random_flip(self, img, seg):
        p = torch.rand(2)
        if p[0] < self.p_flip:
            if p[1] < 0.5:
                img = trsf.hflip(img)
                seg = trsf.hflip(seg)
            else:
                img = trsf.vflip(img)
                seg = trsf.vflip(seg)
        return img, seg

    def random_affine(self, img, seg):
        p = torch.rand(8)
        rotate = 0
        if p[0] < self.p_rotate:
            if p[1] < 0.5:
                rotate += p[1] * self.range_rotate * 2
            else:
                rotate += (p[1] - 1) * 2 * self.range_rotate

        scale = 1
        if p[2] < self.p_scale:
            if p[3] < 0.5:
                scale += p[3] * self.range_scale * 2
            else:
                scale += (p[3] - 1) * self.range_scale * 2

        shear_x = 0
        shear_y = 0
        if p[4] < self.p_shear:
            if p[5] < 0.5:
                shear_x += p[6] * self.range_shear * 2
            else:
                shear_x += (p[6] - 1) * self.range_shear * 2
        if p[6] < self.p_shear:
            if p[7] < 0.5:
                shear_y += p[7] * self.range_shear * 2
            else:
                shear_y += (p[7] - 1) * self.range_shear * 2

        if isinstance(scale, torch.Tensor):
            scale = scale.item()
        if isinstance(rotate, torch.Tensor):
            rotate = rotate.item()
        if isinstance(shear_x, torch.Tensor):
            shear_x = shear_x.item()
        if isinstance(shear_y, torch.Tensor):
            shear_y = shear_y.item()

        img = trsf.affine(
            img,
            rotate,
            [0, 0],
            scale,
            [shear_x, shear_y],
            trsf.InterpolationMode.BILINEAR,
        )
        seg = trsf.affine(
            seg,
            rotate,
            [0, 0],
            scale,
            [shear_x, shear_y],
            trsf.InterpolationMode.NEAREST,
        )
        return img, seg

    def random_crop(self, img, seg):
        if len(img.shape) == 4:
            _, _, w, h = img.shape
        else:
            _, w, h = img.shape
        x = torch.randint(0, w, size=(1,))
        y = torch.randint(0, h, size=(1,))
        d = self.crop_size // 2
        x = min(x, w - d)
        y = min(y, h - d)
        img = trsf.crop(img, x, y, self.crop_size, self.crop_size)
        seg = trsf.crop(seg, x, y, self.crop_size, self.crop_size)
        return img, seg

    def __call__(self, img, seg, **kwds: Any) -> Any:
        img, seg = self.normalize(img, seg)
        img, seg = self.random_flip(img, seg)
        img, seg = self.random_affine(img, seg)
        img, seg = self.random_crop(img, seg)
        return img, seg


class AugmentTest(Augment):
    def __init__(self, crop_size=256) -> None:
        super().__init__(crop_size)

    def __call__(self, img, seg, **kwds: Any) -> Any:
        img, seg = self.normalize(img, seg)
        return img, seg


def save_state(location, epoch, model, optimizer, **kwargs):
    assert "current_epoch" not in kwargs.keys()
    if not os.path.exists(location):
        os.makedirs(location)
    kwargs["current_epoch"] = epoch
    torch.save(kwargs, os.path.join(location, "train.state"))
    torch.save(model.state_dict(), os.path.join(location, "model.state"))
    torch.save(optimizer.state_dict(), os.path.join(location, "optim.state"))


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--epochs", type=int, default=300)
    argparser.add_argument("--lr", type=float, default=0.0003)
    argparser.add_argument("--batch", type=int, default=16)
    argparser.add_argument("--resume", action="store_true")
    argparser.add_argument("--gpu", type=int, default=-1)
    argparser.add_argument("--partation", action="store_true")
    argparser.add_argument("--runs", type=str, default="/tmp/yeruo/runs")
    argparser.add_argument("--fold", type=int, default=1)
    argparser.add_argument("--use-all", action="store_true")
    argparser.add_argument("--best-model", action="store_true")
    argparser.add_argument("--inference", action="store_true")
    args = argparser.parse_args()

    if args.fold == 0:
        dir_runs = args.runs
    else:
        dir_runs = os.path.join(args.runs, "cross", f"fold-{args.fold}")
    dir_partation = args.runs
    dir_latest = dir_runs
    dir_best = os.path.join(dir_runs, "best")
    dir_segout = os.path.join(dir_runs, "segmentation")
    file_logging = os.path.join(dir_runs, "latest_run.log")

    label_names = ["background", "CSF"]
    n_labels = len(label_names)

    check_and_make(args.runs)
    check_and_make(dir_best)
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
        indexs = torch.linspace(0, N - 1, N, dtype=torch.int)
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
                os.path.join(dir_partation, f"fold-{i+1}-train.csv")
            )
            pds.DataFrame(val_set).to_csv(
                os.path.join(dir_partation, f"fold-{i+1}-test.csv")
            )

    logging.info(f"Loading crossover experiment: fold {args.fold}")
    tb_train = pds.read_csv(os.path.join(dir_partation, f"fold-{args.fold}-train.csv"))
    tb_test = pds.read_csv(os.path.join(dir_partation, f"fold-{args.fold}-test.csv"))

    logging.info("Loading model, optimizer, loss function...")
    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=n_labels,
        channels=[32, 64, 64, 128],
        strides=[1, 1, 1],
    )
    loss_func = DiceFocalLoss(to_onehot_y=True)
    scaler = GradScaler()
    watchdog = WatchDog(sniffer=CSFSniffer(label_names))

    augment_train = Augment(crop_size=512)
    augment_test = AugmentTest(crop_size=512)

    logging.info(f"Uploading to device {device}")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.resume:
        cpu = torch.device("cpu")
        if args.best_model:
            logging.info("recover from best model.")
            target_dir = dir_best
        else:
            logging.info("recover from latest model.")
            target_dir = dir_latest
        # model
        state_model = torch.load(
            os.path.join(target_dir, "model.state"), map_location=device
        )
        model.load_state_dict(state_model)
        state_train = torch.load(
            os.path.join(target_dir, "train.state"), map_location=device
        )
        state_optimizer = torch.load(
            os.path.join(target_dir, "optim.state"), map_location=device
        )
        start = state_train["current_epoch"] + 1
        optimizer.load_state_dict(state_optimizer)
        watchdog.load_state(state_train["watchdog"])

    else:
        start = 0

    if not args.inference:
        logging.info(f"Loading Train Dataset...")
        ds_train = CT2DDataset(ds, tb_train, args.use_all)
        logging.info(f"Loading Test Dataset...")
        ds_test = CT2DDataset(ds, tb_test)
        logging.info(f"Traing from epoch {start+1}...")
        counter = Counter(len(ds_train), start=start)
        ds_train = ConcatDataset(
            [
                ShuffleDataset(ds_train, augment_train)
                for _ in range(args.epochs - start)
            ]
        )
        loader_train = tqdm.tqdm(DataLoader(ds_train, batch_size=args.batch))
        for img, seg in loader_train:
            B, C, W, H = img.shape
            img = img.reshape(B * C, 1, W, H).to(device)
            seg = seg.reshape(B * C, 1, W, H).to(device)
            with autocast():
                out = model(img)
                loss = loss_func(out, seg)
            gradient_backward(loss, model, scaler, optimizer)
            loader_train.set_description_str(
                f"Epoch {counter.epoch+1} >>> loss:{loss.item()}"
            )
            watchdog.catch(loss, out, seg)
            counter.count(seg.shape[0])
            if counter.enough():
                for img, seg in DataLoader(ds_test, batch_size=args.batch):
                    with torch.no_grad():
                        B, C, W, H = img.shape
                        img = img.reshape(B * C, 1, W, H).to(device)
                        seg = seg.reshape(B * C, 1, W, H).to(device)
                        img, seg = augment_test(img, seg)
                        with autocast():
                            out = model(img)
                            loss = loss_func(out, seg)
                        watchdog.catch(loss, out, seg, mode="validate")
                current = watchdog.step()
                logging.info(str(current)[1:-1])
                if watchdog.happy():
                    logging.info(f"[Better weight]:{current}")
                    save_state(
                        dir_best,
                        counter.epoch,
                        model,
                        optimizer,
                        watchdog=watchdog.state_dict(),
                    )
                else:
                    save_state(
                        dir_latest,
                        counter.epoch,
                        model,
                        optimizer,
                        watchdog=watchdog.state_dict(),
                    )
                counter.step()
                logging.info(f"Epoch {counter.epoch} finished.")

    else:
        logging.info(f"Loading Test Dataset...")
        with torch.no_grad():
            for sid in tb_test["SubjectID"]:
                img = ds.img_file(sid)
                nii = nib.load(img)
                data = nii.get_fdata()
                img: torch.Tensor = torch.from_numpy(data)
                img = img.permute(2, 0, 1)
                img = img.unsqueeze_(dim=1).float()
                res = []
                for im in torch.split(img, args.batch, dim=0):
                    im = im.to(device)
                    im, _ = augment_test(im, None)
                    out = model(im)
                    seg = out.argmax(dim=1)
                    res.append(seg.cpu().numpy())
                res = np.concatenate(res, axis=0)
                res = res.transpose(1, 2, 0)
                print(data.shape, res.shape)
                seg = nib.Nifti1Image(res.astype(np.uint8), affine=nii._affine)
                nib.save(seg, os.path.join(dir_segout, f"{sid}.nii.gz"))

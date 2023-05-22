import logging
import os
import time
from argparse import ArgumentParser
from typing import *

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchio as tio
import torchvision.transforms.functional as trsf
import tqdm
from monai.networks.nets.unetr import UNETR
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

from ABrain.dataset import CTCSF, OurDataset, Subset
from ABrain.evaluation import dice
from ABrain.inference import UNet3DGridAggregator, UNet3DGridPatch
from ABrain.modelzoo.losses import DiceLoss2D, DiceLoss3D
from ABrain.trainer.watchdog import Sniffer, WatchDog
from ABrain.trainer.writer import NIfTIWriter


class Augment(object):
    def __init__(self) -> None:
        self.p_flip = 0.3
        self.p_scale = 0.3
        self.p_rotate = 0.3
        self.p_shear = 0.3

        self.range_scale = 0.3
        self.range_rotate = 30
        self.range_shear = 10
        self.crop_size = 256

    def normalize(self, img: torch.Tensor, seg):
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img)
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img)
        img = torch.clamp_(img, -10, 244)
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
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, img, seg, **kwds: Any) -> Any:
        img, seg = self.normalize(img, seg)
        return img, seg


def make_hd5(ds: OurDataset, path: str):
    img_pool = []
    seg_pool = []
    for sid in ds.sids:
        img = ds.img_file(sid)
        seg = ds.seg_file(sid)
        subject = Subject(img=ScalarImage(img), seg=LabelMap(seg))
        try:
            subject.check_consistent_spatial_shape()
        except:
            print(subject["img"].shape, subject["seg"].shape, sid)
            continue
        img = subject["img"][DATA][:, :, :, 2:-2]
        seg = subject["seg"][DATA][:, :, :, 2:-2]
        img_pool.append(img)
        seg_pool.append(seg)
    # 1,512,512,n
    img_pool = torch.cat(img_pool, dim=3).permute(3, 0, 1, 2)
    seg_pool = torch.cat(seg_pool, dim=3).permute(3, 0, 1, 2)
    with h5py.File(os.path.join(path, "CTCSF.hd5"), "w") as h5:
        h5.create_dataset("img", data=img_pool, compression="gzip")
        h5.create_dataset("seg", data=seg_pool, compression="gzip")


class H5Dataset(Dataset):
    def __init__(self, path: str, transforms: Optional[Callable] = None) -> None:
        super().__init__()
        self.path = path
        self.transforms = transforms

    def try_open(self):
        if not hasattr(self, "h5file"):
            self.h5file = h5py.File(self.path)

    def __getitem__(self, index) -> Any:
        self.try_open()
        img = self.h5file["img"][index, :, :, :]
        seg = self.h5file["seg"][index, :, :, :]
        if self.transforms:
            img, seg = self.transforms(img, seg)
        return img, seg

    def __len__(self):
        self.try_open()
        return self.h5file["img"].shape[0]


class CT2DDataset(Dataset):
    def __init__(self, ds: OurDataset, transforms=None) -> None:
        super().__init__()
        self.ds = ds
        self.n = len(self.ds.sids)
        self.transform = transforms
        img_pool = []
        seg_pool = []
        for sid in ds.sids:
            img = ds.img_file(sid)
            seg = ds.seg_file(sid)
            subject = Subject(img=ScalarImage(img), seg=LabelMap(seg))
            try:
                subject.check_consistent_spatial_shape()
            except:
                print(
                    f'Wrong size ({sid}): img{subject["img"].shape}, seg{subject["seg"].shape}'
                )
                continue
            img = subject["img"][DATA][:, :, :, 2:-2]
            seg = subject["seg"][DATA][:, :, :, 2:-2]
            img_pool.append(img)  # 1,512,512,n
            seg_pool.append(seg)

        self.img_pool = torch.cat(img_pool, dim=3).permute(3, 0, 1, 2)
        self.seg_pool = torch.cat(seg_pool, dim=3).permute(3, 0, 1, 2)

    def make_tensor_dataset(self):
        img = self.img_pool
        seg = self.seg_pool
        seg = self.seg_pool.squeeze(dim=1)
        seg[seg == 3] = 1
        return TensorDataset(img, seg)

    def __getitem__(self, index) -> Any:
        img = self.img_pool[index, :, :, :]
        seg = self.seg_pool[index, :, :, :]
        if self.transform:
            img, seg = self.transform(img, seg)
        return dict(img=img, seg=seg)

    def __len__(self):
        return self.img_pool.shape[0]


class Timer(object):
    def __init__(self) -> None:
        self._msg = ""

    def __enter__(self):
        self.t1 = time.time()
        self._msg = ""
        return self

    def msg(self, msg: str):
        self._msg = msg

    def __exit__(self, *args):
        t = time.time() - self.t1
        logging.info(f"{self._msg} {t} s.")


class CT2Test(Dataset):
    def __init__(self, ds: OurDataset, transforms=None) -> None:
        super().__init__()
        self.sids = []
        self.transforms = transforms
        for sid in ds.sids:
            img = ds.img_file(sid)
            seg = ds.seg_file(sid)
            subject = Subject(img=ScalarImage(img), seg=LabelMap(seg), name=sid)
            try:
                subject.check_consistent_spatial_shape()
            except:
                continue
            self.sids.append(sid)
        self.ds = ds

    def __getitem__(self, index):
        sid = self.sids[index]
        img = self.ds.img_file(sid)
        seg = self.ds.seg_file(sid)
        subject = Subject(img=ScalarImage(img), seg=LabelMap(seg), name=sid)
        if self.transforms:
            subject = self.transforms(img, seg)
        return subject

    def __len__(self):
        return len(self.sids)


def parse_data(data: Tuple[torch.Tensor, torch.Tensor], device: torch.device):
    img, seg = data
    img, seg = img.float(), seg.long()
    return img.to(device), seg.to(device)


class CSF_Sniffer(Sniffer):
    def __init__(self, label_names):
        self.labels = label_names

    def sniffing(self, pred, target):
        d = dice(pred.argmax(dim=1), target, n_labels=len(label_names))
        return {k: v for k, v in zip(self.labels, d)}

    def is_better(self, current, history):
        return current["CSF"] > history["CSF"]


def gradient_backward(loss, model, scaler, optimizer):
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    nn.utils.clip_grad.clip_grad_value_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()


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
    argparser.add_argument("--batch-size", type=int, default=32)
    argparser.add_argument("--prepare-dataset", action="store_true")
    argparser.add_argument("--hd5", action="store_true")
    argparser.add_argument("--resume", action="store_true")
    argparser.add_argument("--gpu-id", type=int, default=-1)
    argparser.add_argument("--partation", action="store_true")
    argparser.add_argument("--runs", type=str, default="/tmp/yeruo/runs")
    argparser.add_argument("--fold", type=int, default=1)
    argparser.add_argument("--best-model", action="store_true")

    args = argparser.parse_args()

    file_partations = os.path.join(args.runs, "partation.pt")
    file_hd5 = os.path.join("/dev/shm", "CTCSF.hd5")
    file_log = os.path.join("latest_run.log")
    dir_best = os.path.join(args.runs, "best")
    dir_latest = args.runs
    dir_segout = os.path.join(args.runs, "segmentation")

    handler = logging.FileHandler(file_log, mode="a+")
    logging.getLogger().addHandler(handler)

    for each in dir(args):
        if not each.startswith("__"):
            logging.info(f"{each}-->{getattr(args,each)}")
    if not os.path.exists(args.runs):
        os.makedirs(args.runs)

    timer = Timer()

    if args.gpu_id < 0:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu_id}")

    logging.info(f"Using Device {device.type}:{device.index}")

    logging.info(f"Loggings in {file_log}")

    ds = CTCSF("5.0mm")
    label_names = ["background", "CSF"]
    n_labels = len(label_names)

    if args.partation:
        N = len(ds)
        indexs = torch.linspace(0, N - 1, N, dtype=int)
        cross5 = random_split(indexs, [0.2, 0.2, 0.2, 0.2, 0.2])
        folds = {}
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
            train_set = torch.tensor(train_set)
            val_set = torch.tensor(val_set)
            folds[f"fold {i+1}"] = {"train": train_set, "test": val_set}
        logging.info(
            f"{N} samples is divided in {5} folds, the the partation file is in {file_partations}"
        )
        torch.save(folds, file_partations)

    if args.fold == 0:
        ds_train = ds
        ds_test = None
        logging.info("Training with all the samples.")
    else:
        partation = torch.load(os.path.join(args.runs, "partation.pt"))
        ids_train: torch.Tensor = partation[f"fold {args.fold}"]["train"]
        ids_test: torch.Tensor = partation[f"fold {args.fold}"]["test"]
        ds_train = Subset(ds, ids_train)
        ds_test = Subset(ds, ids_test)
        logging.info(
            f"Training over fold {args.fold}, {len(ids_train)} for train, {len(ids_test)} for test."
        )

    logging.info("Loading dataset...")
    with timer as t:
        t.msg("Load dataset token")
        ds_train = CT2DDataset(ds_train)
        ds_train = ds_train.make_tensor_dataset()
        ds_test = CT2Test(ds_test)

    logging.info("Initialize model, augmenter, watchdog, optimizer, loss function...")
    aug = Augment()
    watchdog = WatchDog(CSF_Sniffer(label_names))
    model = UNETR(
        in_channels=1, out_channels=n_labels, img_size=256, spatial_dims=2
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler()
    # loss_func = torch.nn.CrossEntropyLoss()
    loss_func = DiceLoss2D()
    loss_func_test = DiceLoss3D()
    start = 0

    if args.resume:
        if args.best_model:
            logging.info("Loading `best` snapshoot...")
            target_dir = dir_best
        else:
            logging.info("Loading `latest` snapshoot...")
            target_dir = dir_latest
        # model
        state_model = torch.load(os.path.join(target_dir, "model.state"))
        model.load_state_dict(state_model)
        state_train = torch.load(os.path.join(target_dir, "train.state"))
        state_optimizer = torch.load(os.path.join(target_dir, "optim.state"))
        start = state_train["current_epoch"] + 1
        optimizer.load_state_dict(state_optimizer)
        watchdog.load_state(state_train["watchdog"])

    n = len(ds_train)
    epochs = args.epochs - start
    logging.info(f"{epochs} epochs need to be done, current is Epoch {start}")
    ds_train = ConcatDataset([ds_train] * epochs)

    iters = 0
    epoch = 0

    loader = tqdm.tqdm(
        DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    )
    writer = NIfTIWriter(dir_segout)
    for data in loader:
        img, seg = parse_data(data, device)
        img, seg = aug(img, seg)
        with autocast():
            out = model(img)
            loss = loss_func(out, seg)
        gradient_backward(loss, model, scaler, optimizer)
        loader.set_description_str(f"loss:{loss.item()}")
        watchdog.catch(loss, out, seg)
        iters += img.shape[0]
        if iters >= n:
            model.eval()
            for subject in ds_test:
                name = subject["name"]
                seg = subject["seg"][DATA].long().to(device)
                seg[seg == 3] = 1
                patchs = UNet3DGridPatch(subject, (256, 256, 1), (256, 256, 1))
                aggregator = UNet3DGridAggregator(subject)
                with torch.no_grad():
                    for img, locations in DataLoader(
                        patchs, batch_size=args.batch_size
                    ):
                        img: torch.Tensor = img.float().to(device)
                        a, c, w, h, d = img.shape
                        img = img.permute(0, 4, 1, 2, 3)  # a,c,d,w,h
                        img = img.reshape(a * d, c, w, h)
                        logits = model(img.to(device))
                        logits = logits.reshape(a, d, -1, w, h)
                        logits = logits.permute(0, 2, 3, 4, 1)
                        aggregator.add_batch(logits, locations)
                output = aggregator.get_output_tensor(cpu=False)
                writer.save(
                    (name, subject["img"].affine), output.argmax(dim=1).squeeze()
                )
                output.unsqueeze_(dim=0)
                # seg.unsqueeze_(dim=0)
                loss = loss_func_test(output, seg)
                watchdog.catch(loss, output, seg, mode="validate")
                logging.info(f"loss of {name} is {loss.item()}")
            model.train()
            epoch += 1
            iters -= n
            current = str(watchdog.step())[1:-1]
            if watchdog.happy():
                logging.info(f"[Better weight]:{current}")
                save_state(
                    dir_best, epoch, model, optimizer, watchdog=watchdog.state_dict()
                )
            else:
                save_state(
                    dir_latest, epoch, model, optimizer, watchdog=watchdog.state_dict()
                )

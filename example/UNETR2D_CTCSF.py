import os
import argparse
import logging
from typing import *


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.utils.data as data
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torchio import DATA, Subject, LabelMap, ScalarImage
import torchio as tio
import pandas as pds

import monai.networks.nets as nets

from ABrain.modelzoo.losses import DiceLoss3D
from ABrain.dataset import OurDataset, Subset, CTCSF, DatasetWapper
from ABrain.transforms import ZNormalization, HistogramStandardization, ToCanonical
from ABrain.transforms import Pad, Crop
from ABrain.transforms import (
    RandomFlip,
    RandomAffine,
    RandomGhosting,
    RandomElasticDeformation,
    RandomGamma,
    Pad,
)
from ABrain.transforms import Compose
from ABrain.trainer.watchdog import WatchDog, Sniffer
from ABrain.evaluation import (
    dice,
    VoxelMethods,
    VoxelMetrics,
    HausdorffDistance95,
    ComposeAnalyzer,
)
from ABrain.inference import UNet3DGridPatch, UNet3DGridAggregator


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--epochs", type=int, default=300, help="训练回合数")
arg_parser.add_argument("--resume", action="store_true", help="是否接上次训练")
arg_parser.add_argument("--test", action="store_true", help="在测试集上执行测试")
arg_parser.add_argument("--lr", type=float, default=3e-4, help="学习率")
arg_parser.add_argument("--batch-size", type=int, default=128, help="Mini-Batch大小")
arg_parser.add_argument("--patch-size", nargs=2, type=int, required=True, help="训练块大小")
arg_parser.add_argument(
    "--runs", type=str, default="tmp/yeruo/runs", help="训练参数及过程记录存放的文件夹"
)
# arg_parser.add_argument('--label_names',)
arg_parser.add_argument("--gpu-id", type=int, default=1, help="GPU ID")
arg_parser.add_argument("--inference", type=str, default="./inference", help="预测结果存放位置")
arg_parser.add_argument("--best-model", action="store_true", help="从验证集上最好的那套参数开始工作")
arg_parser.add_argument(
    "--compile-mode", help="Torch 2.0 compile mode", type=str, default="default"
)
arg_parser.add_argument("--label-smooth", help="标签平滑", type=float, default=0.0)

args = arg_parser.parse_args()

resume = args.resume
epochs = args.epochs
lr = args.lr
batch_size = args.batch_size
patch_size = tuple(args.patch_size)
# padding from edge = (128-36)/2 = 46
label_names = ["BG", "CSF"]
# label_names = args.label_names

run_dir = args.runs
best_dir = os.path.join(run_dir, "best")
logging.basicConfig(level=logging.INFO)
if not os.path.exists(run_dir):
    os.makedirs(run_dir)
log_path = os.path.join(run_dir, "latest_run.log")
handler = logging.FileHandler(log_path, mode="a+")
logging.getLogger().addHandler(handler)

if args.gpu_id < 0:
    device = torch.device("cpu")
else:
    device = torch.device(f"cuda:{args.gpu_id}")
loss_func = nn.CrossEntropyLoss(label_smoothing=args.label_smooth)
# loss_func = DiceLoss3D()

preprocess = ToCanonical()
normalization = ZNormalization()
# padding = Pad([46, 46, 46], padding_mode="reflect")
augmentations = Compose(
    [
        RandomFlip(axes=(0, 1, 2), flip_probability=0.5),
        RandomAffine(
            scales=0.2, degrees=25, center="image", default_pad_value="minimum"
        ),
        RandomElasticDeformation(),
        # RandomGhosting(),
        RandomGamma(),
    ]
)


logging.info(f"Using `3D U-Net` as the base model.")
model = nets.unetr.UNETR(
    in_channels=1, out_channels=len(label_names), img_size=(512, 512), spatial_dims=2
)

logging.info(f"[Compile]: mode --> {args.compile_mode}")
model = torch.compile(model, mode=args.compile_mode)


class CT2DDataset(Dataset):
    def __init__(self, ds: OurDataset, index_rand, transforms=None) -> None:
        super().__init__()
        self.ds = ds
        self.index_rand = index_rand
        self.n = len(self.ds.sids)
        self.transfroms = transforms

    def __getitem__(self, index) -> Any:
        rid = self.index_rand[index]
        sid = self.ds.sids[rid % self.n]
        img = self.ds.img_file(sid)
        seg = self.ds.seg_file(sid)
        subject = Subject(img=ScalarImage(img), seg=LabelMap(seg))
        if self.transfroms:
            subject = self.transfroms(subject)
        _, _, _, d = subject.shape
        indexd = rid % d
        slice_img = subject["img"][DATA][:, :, :, indexd]
        slice_seg = subject["seg"][DATA][:, :, :, indexd]
        return dict(img=slice_img, seg=slice_seg)

    def __len__(self):
        return len(self.index_rand)


class CT2Test(Dataset):
    def __init__(self, ds: OurDataset, transforms) -> None:
        super().__init__()
        self.subject_id = []
        self.slice_id = []
        self.ds = ds
        self.transforms = transforms
        for i, sid in enumerate(ds.sids):
            img = ds.img_file(sid)
            nii = ScalarImage(img)
            _, _, _, d = nii.shape
            self.subject_id.extend([i] * d)
            self.slice_id.extend(range(d))
        self.current = ""

    def __getitem__(self, index):
        subject_id = self.subject_id[index]
        sid = self.ds.sids[subject_id]
        if self.current != sid:
            img = self.ds.img_file(sid)
            seg = self.ds.seg_file(sid)
            subject = Subject(img=ScalarImage(img), seg=LabelMap(seg))
            if self.transforms:
                subject = self.transforms(subject)
            self.subject = subject
            self.current = sid
        slice_id = self.slice_id[index]
        img = self.subject["img"][DATA]
        seg = self.subject["seg"][DATA]
        return dict(img=img[:, :, :, slice_id], seg=seg[:, :, :, slice_id])

    def __len__(self):
        return len(self.slice_id)


class CSF_Sniffer(Sniffer):
    def __init__(self, label_names):
        self.labels = label_names

    def sniffing(self, pred, target):
        d = dice(pred.argmax(dim=1), target, n_labels=len(label_names))
        return {k: v for k, v in zip(self.labels, d)}

    def is_better(self, current, history):
        return current["CSF"] > history["CSF"]


watchdog = WatchDog(CSF_Sniffer(label_names))

if resume or args.test:
    if args.best_model:
        logging.info("Loading `best` snapshoot...")
        target_dir = best_dir
    else:
        logging.info("Loading `latest` snapshoot...")
        target_dir = run_dir
    # model
    state_model = torch.load(os.path.join(target_dir, "model.state"))
    model.load_state_dict(state_model)
    model.to(device)

    # training..
    if not args.test:
        state_train = torch.load(os.path.join(target_dir, "train.state"))
        state_optimizer = torch.load(os.path.join(target_dir, "optim.state"))
        start = state_train["current_epoch"] + 1
        optimizer = optim.Adam(model.parameters(), lr=lr)
        optimizer.load_state_dict(state_optimizer)
        watchdog.load_state(state_train["watchdog"])
    else:
        start = 0
        optimizer = None
    state_model = None
    state_train = None
    state_optimizer = None
    # torch.cuda.empty_cache()

else:
    logging.info("Training from scratch...")
    start = 0
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

scaler = GradScaler()


def parse_data(package, device):
    img = package["img"]
    seg = package["seg"]
    target = seg.long()
    return img.to(device), target.to(device)


def gradient_backward(loss, scaler, optimizer):
    scaler.scale(loss).backward()
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


if not args.test:
    # prepare train and val dataset

    trans_train = Compose([preprocess, augmentations, normalization])
    trans_val = Compose([preprocess, normalization])

    ids_train = list(range(10))
    ids_val = list(range(80, 96))
    rand_ids = torch.randint(123456789, (10000,))
    ds = CTCSF(resolution="5.0mm")
    ds_train = Subset(ds, ids_train)
    ds_train = CT2DDataset(ds_train, rand_ids, trans_train)
    ds_val = Subset(ds, ids_val)
    ds_val = CT2Test(ds_val, trans_val)

    n_train = len(ds_train)
    n_val = len(ds_val)
    logging.info(f"Using `CTCSF`, {n_train} train samples, {n_val} for validation.")

    loader_train = data.DataLoader(ds_train, batch_size, num_workers=4)
    loader_val = data.DataLoader(ds_val, batch_size, num_workers=1)

    logging.info(f"Start training from round {start+1}, total {epochs} rounds.")
    for epoch in range(start, epochs):
        model.train()
        for each in loader_train:
            input, target = parse_data(each, device)
            with autocast():
                out = model(input)
                loss = loss_func(out, target)
            gradient_backward(loss, scaler, optimizer)
            watchdog.catch(loss, out, target)
        model.eval()
        for each in loader_val:
            input, target = parse_data(each, device)
            with torch.no_grad():
                out = model(input)
                loss = loss_func(out, target)
                watchdog.catch(loss, out, target, mode="validate")
        current = str(watchdog.step())[1:-1]
        logging.info(f"[Epoch {epoch+1}]:{current}")
        if watchdog.happy():
            logging.info(f"[Better weight]:{current}")
            save_state(
                best_dir, epoch, model, optimizer, watchdog=watchdog.state_dict()
            )
        save_state(run_dir, epoch, model, optimizer, watchdog=watchdog.state_dict())


def parse_data_test(package, device):
    img = package["img"][DATA]
    locations = package[tio.LOCATION]
    return img.to(device), locations


if args.test:
    pass
    # dataset = CTCSF()
    # dataset = Subset(dataset, list(range(80, 96)))
    # dataset = DatasetWapper(dataset)
    # result = []
    # test_func = ComposeAnalyzer(
    #     HausdorffDistance95(label_names), VoxelMetrics(VoxelMethods.keys(), label_names)
    # )
    # for each in dataset:
    #     with torch.no_grad():
    #         subject = trans_test(each)
    #         seg = subject["seg"][DATA]
    #         name = subject["name"]
    #         ds = UNet3DGridPatch(subject, patch_size, out_size)
    #         aggregator = UNet3DGridAggregator(subject)
    #         for img, locations in data.DataLoader(ds, batch_size=args.batch_size):
    #             logits = model(img.to(device))
    #             aggregator.add_batch(logits, locations)
    #         output = aggregator.get_output_tensor()
    #
    #         metrics = test_func(
    #             Subject(
    #                 name=name,
    #                 true=subject["seg"],
    #                 pred=LabelMap(tensor=output.argmax(dim=0, keepdim=True)),
    #             )
    #         )
    #         msg = str(metrics)[1:-1]
    #         logging.info(f"[Metrics]: {msg}")
    #         torch.save(output, name)
    #         result.append(metrics)
    # result = pds.DataFrame(result)
    # result.to_csv("test.csv")

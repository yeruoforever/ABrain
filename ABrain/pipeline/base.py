import argparse
import logging
import os
import random
from typing import Optional
from importlib.util import spec_from_file_location, module_from_spec

import numpy as np
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Dataset
from torchio.transforms import Compose, OneOf

from ..dataset.base import DatasetWapper
from ..factory import *
from ..trainer.segmentation import SegmentationTrainer
from ..trainer.watchdog import Sniffer


def mkdirs(dirs: str):
    if not os.path.exists(dirs):
        os.makedirs(dirs)


class Experiment(object):
    def __init__(self, name: str, database: str) -> None:
        self.name = name

        self.database = os.path.join(database, name)
        self.dir_runs = os.path.join(self.database, "runs")
        self.dir_results = os.path.join(self.database, "results")
        self.dir_inferences = os.path.join(self.database, "inferences")

        mkdirs(self.dir_runs)
        mkdirs(self.dir_results)
        mkdirs(self.dir_inferences)

        path = os.path.join(self.database, "config.py")
        spec = spec_from_file_location("config", path)
        if spec is None:
            raise FileNotFoundError(f"`config.py` not find in {path} ")
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        self.config = module


def better_print(data: dict):
    return ", ".join(k+":%.4f" % v for k, v in data.items())


class TrainEnegine(object):
    def __init__(self, path, sniffer=Sniffer) -> None:
        self.init_argsparser()
        self.parse_args(path)
        self.set_seed()
        self.make_trainer(sniffer)
        pass

    def init_argsparser(self):
        parser = argparse.ArgumentParser(description="")

        parser.add_argument("experiment", type=str,
                            help="Experiment name.(or Task name.)")

        parser.add_argument("-r", "--resume", action='store_true')
        parser.add_argument("-t", "--train", action='store_true')
        parser.add_argument("-v", "--validate", action='store_true')
        parser.add_argument("-e", "--test", action='store_true')
        parser.add_argument("-i", "--inference", action='store_true')

        parser.add_argument("--gpu_id", type=int, default=-1)
        parser.add_argument("--num_worker", type=int, default=8)

        parser.add_argument("--use_latest_model", action='store_true')

        self.parser = parser

    def parse_args(self, path):
        args = self.parser.parse_args()
        if args.gpu_id < 0:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device("cuda", args.gpu_id)
        self.experiment = Experiment(args.experiment, path)
        self.config = self.experiment.config
        self.args = args
        self.epoch = 0

    def set_seed(self):
        seed = self.config.seed
        torch.set_printoptions(precision=5)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def make_trainer(self, SNF):
        model = self.config.model.to(self.device)
        n_labels = self.config.n_labels
        label_names = self.config.label_names
        sniffer = SNF(label_names)
        deep_superwised = self.config.deep_supervise
        trainer = SegmentationTrainer(
            model, n_labels, sniffer, self.device,
            deep_supervise=deep_superwised
        )
        self.model = model
        self.trainer = trainer

    def save_state(self, optimizer, scheduler, loss_func, label: str):
        state = {
            "scheduler": scheduler.state_dict(),
            "loss_func": loss_func.state_dict(),
            "trainer": self.trainer.state_dict(),
            "epoch": self.epoch
        }
        path = os.path.join(self.experiment.dir_runs,
                            f"training.{label}.state")
        torch.save(state, path)
        path = os.path.join(self.experiment.dir_runs, f"model.{label}.state")
        torch.save(self.model.state_dict(), path)
        path = os.path.join(self.experiment.dir_runs,
                            f"optimizer.{label}.state")
        torch.save(optimizer.state_dict(), path)

    def load_model(self, label: str):
        path = os.path.join(self.experiment.dir_runs, f"model.{label}.state")
        state = torch.load(path,map_location=self.device)
        self.trainer.reload_model(state)

    def load_state(self, optimizer, scheduler: _LRScheduler, loss_func, label: str):
        logging.info(f"loading train state from `{label}` version...")
        path = os.path.join(self.experiment.dir_runs,
                            f"training.{label}.state")
        state = torch.load(path)
        self.trainer.load_state(state["trainer"])
        scheduler.load_state_dict(state["scheduler"])
        loss_func.load_state_dict(state["loss_func"])
        self.epoch = state["epoch"]

        try:
            path = os.path.join(self.experiment.dir_runs,
                                f"optimizer.{label}.state")
            state = torch.load(path)
        except:
            state = state["optimizer"]
        optimizer.load_state_dict(state)

        path = os.path.join(self.experiment.dir_runs, f"model.{label}.state")
        state = torch.load(path)
        self.model.load_state_dict(state)

        lr = scheduler.get_last_lr()
        logging.info(f"current epoch is {self.epoch}, learn rate is {lr}.")
        metrices = better_print(self.trainer.watcher.current())
        logging.info(f"current metrices is {metrices}.")

    def get_dataloader(self, mode: str):
        if mode == 'validation' and self.args.validate is False:
            return None
        batch_size = self.config.batch_size
        n_workers = self.args.num_worker
        if mode in ['test','inference']:
            batch_size=1
        loader_config = {
            "batch_size": batch_size,
            "num_workers": n_workers,
            "shuffle": True,
            "persistent_workers": True
        }
        ds = self.config.datasets[mode]
        ts = Compose(self.config.transforms[mode])
        ds = DatasetWapper(ds, ts).pipeline()
        return DataLoader(ds, **loader_config)

    def train(self):
        dl_train = self.get_dataloader("train")
        dl_val = self.get_dataloader("validation")

        config, trainer = self.config, self.trainer

        optimizer = config.optimizer
        scheduler = config.scheduler
        loss_func = config.loss_fun

        if self.args.resume:
            self.load_state(optimizer, scheduler, loss_func, 'latest')

        while self.epoch < config.epochs:
            self.epoch += 1
            msg = f"Epoch {self.epoch}"
            trainer.train(dl_train, loss_func, optimizer, info=msg)
            if self.args.validate:
                trainer.validate(dl_val, loss_func, info=msg)
            current = trainer.epoch_step()
            scheduler.step()
            logging.info(f"[{msg}]: {better_print(current)}")
            logging.info(f"[{msg}]: current lr is {scheduler.get_last_lr()}")
            if trainer.watcher.happy():
                metrices = better_print(trainer.watcher.current())
                logging.info(f"[better model]: {metrices}")
                self.save_state(optimizer, scheduler, loss_func, "best")
            self.save_state(optimizer, scheduler, loss_func, "latest")

    def test(self):
        logging.info("test")
        if self.args.use_latest_model:
            self.load_model("latest")
        else:
            self.load_model("best")
        dl_test = self.get_dataloader("test")
        patch_size = self.config.patch_size
        stride = tuple(int(p*s) for p, s in zip(patch_size, self.config.stride_size))
        logging.info(f"[testing]: patch: {patch_size}, stride: {stride}")
        # self.trainer.inference(dl_test, self.config.writer, patch_size=patch_size, stride=stride)
        self.config.evaluator.run()
        self.config.evaluator.show()
        self.config.evaluator.save(os.path.join(self.experiment.dir_results,'evaluater.csv'))
        

    def inference(self):
        logging.info("inference")
        if self.args.use_latest_model:
            self.load_model("latest")
        else:
            self.load_model("best")
        dl_test = self.get_dataloader("inference")
        dl_test = self.get_dataloader("test")
        patch_size = self.config.patch_size
        stride = tuple(int(p*s) for p, s in zip(patch_size, self.config.stride_size))
        logging.info(f"[testing]: patch: {patch_size}, stride: {stride}")
        self.trainer.inference(dl_test, self.config.writer, patch_size=patch_size, stride=stride)

    def run(self):

        if self.args.train:
            self.train()

        if self.args.test:
            self.test()

        if self.args.inference:
            self.inference()

import os
import re
from typing import List

import numpy as np
import pandas as pds
import scipy
import torch
import torchvision
import tqdm
from torch.utils.data import Dataset

from ..transforms.RGB_PCA import RGBPCA


class ILSVRC2010(object):
    def load_meta(self, database):
        self.database = database
        meta = scipy.io.loadmat(os.path.join(
            database, 'devkit-1.0', 'data', 'meta.mat'))
        synset_name = [str(meta['synsets'][i][0][1][0])
                       for i in range(meta["synsets"].shape[0])]
        synset_id = [meta['synsets'][i][0][0][0][0]
                     for i in range(meta["synsets"].shape[0])]
        synset_words = [meta['synsets'][i][0][2][0]
                        for i in range(meta["synsets"].shape[0])]
        self.vocab = dict(zip(synset_name, synset_id))
        self.keywords = dict(zip(synset_id, synset_words))

    def get_jpeg_list(self, dir: str) -> List[str]:
        jpegs = []
        p = re.compile(".*\.[jJ][pP][eE]?[gG]")

        def list_jpeg(dir):
            for e in os.listdir(dir):
                path = os.path.join(dir, e)
                if os.path.isdir(path):
                    list_jpeg(path)
                elif re.match(p, path) is not None:
                    jpegs.append(path)

        list_jpeg(dir)
        return jpegs

    def map_name_id(self, name: str):
        p = re.compile("[_/]")
        synset_name = re.split(p, name)[-2]
        sid = self.vocab[synset_name]
        return sid


default_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.RandomCrop(224),
    torchvision.transforms.ConvertImageDtype(torch.float32),
    RGBPCA()
])


class ILSVRC2010Train(Dataset, ILSVRC2010):
    def __init__(self, database, transform=default_transform) -> None:
        super().__init__()
        self.load_meta(database)
        dir_train = os.path.join(database, "train")
        record_file = os.path.join(database, "train_labels.csv")
        if not os.path.exists(record_file):
            self.write_file_list(dir_train, record_file)
        record = pds.read_csv(record_file)

        self.jpegs = record["FileName"]
        self.label = record["label"]

        self.transform = transform

    def __len__(self):
        return len(self.jpegs)

    def __getitem__(self, index):
        img = self.jpegs[index]
        lab = self.label[index]-1
        data = torchvision.io.read_image(img)
        data: torch.Tensor = self.transform(data)
        return data, lab

    def write_file_list(self, dir: str, file_name: str):
        jpegs = self.get_jpeg_list(dir)
        targets = []
        for each in tqdm.tqdm(jpegs):
            try:
                img = torchvision.io.read_image(each)
                if img.shape[0] == 3:
                    targets.append(each)
            except KeyboardInterrupt as e:
                raise e
            except RuntimeError as e:
                print(e)
                continue
        labes = np.array(list(map(self.map_name_id, targets)), dtype=int)
        df = pds.DataFrame({
            "FileName": targets,
            "label": labes
        })
        df.to_csv(file_name)


class ILSVRC2010Valitation(Dataset, ILSVRC2010):
    def __init__(self, database, transform=default_transform) -> None:
        super().__init__()
        self.load_meta(database)
        dir_val = os.path.join(database, "val")
        record_file = os.path.join(database, "val_label.csv")

        if not os.path.exists(record_file):
            files = os.listdir(dir_val)
            files.sort()
            record = np.loadtxt(os.path.join(
                database, "devkit-1.0", "data",
                "ILSVRC2010_validation_ground_truth.txt"
            ), dtype=int)

            jpegs = []
            label = []
            for jpg, l in tqdm.tqdm(zip(files, record)):
                p = os.path.join(dir_val, jpg)
                img = torchvision.io.read_image(p)
                if img.shape[0] == 3:
                    jpegs.append(p)
                    label.append(l)
            pds.DataFrame({
                "FileName": jpegs,
                "label": label
            }).to_csv(record_file)
        record = pds.read_csv(record_file)
        self.jpegs = record["FileName"]
        self.label = record["label"]
        self.transform = transform

    def __len__(self):
        return len(self.jpegs)

    def __getitem__(self, index):
        img = self.jpegs[index]
        lab = self.label[index]-1
        data = torchvision.io.read_image(img)
        data: torch.Tensor = self.transform(data)
        return data, lab


class ILSVRC2010Test(Dataset, ILSVRC2010):
    def __init__(self, database, transform=default_transform) -> None:
        super().__init__()
        self.load_meta(database)
        dir_val = os.path.join(database, "test")
        record_file = os.path.join(database, "test_label.csv")

        if not os.path.exists(record_file):
            files = os.listdir(dir_val)
            files.sort()
            record = np.loadtxt(os.path.join(
                database,
                "ILSVRC2010_test_ground_truth.txt"
            ), dtype=int)

            jpegs = []
            label = []
            for jpg, l in tqdm.tqdm(zip(files, record)):
                p = os.path.join(dir_val, jpg)
                img = torchvision.io.read_image(p)
                if img.shape[0] == 3:
                    jpegs.append(p)
                    label.append(l)
            pds.DataFrame({
                "FileName": jpegs,
                "label": label
            }).to_csv(record_file)
        record = pds.read_csv(record_file)
        self.jpegs = record["FileName"]
        self.label = record["label"]
        self.transform = transform

    def __len__(self):
        return len(self.jpegs)

    def __getitem__(self, index):
        img = self.jpegs[index]
        lab = self.label[index]-1
        data = torchvision.io.read_image(img)
        data: torch.Tensor = self.transform(data)
        return data, lab

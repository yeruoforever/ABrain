import toml

from torchio import Subject, ScalarImage, LabelMap
from torchio import Compose
from torchio.transforms.augmentation import RandomTransform

from torch.utils.data import Dataset
from torchdata.datapipes import iter


def read_config(conf_name:str):
    with open("database.toml", "r") as f:
        config = toml.load(f)
    if conf_name in config.keys():
        return config[conf_name]
    else:
        datasets = config["Dataset"]
        n = len(datasets)
        keys = {datasets[i]["name"]:i for i in range(n)}
        if conf_name in keys:
            return datasets[keys[conf_name]]
        else:
            raise KeyError(f"{conf_name} is not a dataset.")



class OurDataset(object):
    def __init__(
        self,
        database,
        name:str,
        has_seg: bool = False,
        has_label: bool = False,
        has_info: bool = False
    ) -> None:
        super().__init__()
        self.database = database
        self.name = name
        self.sids = self.get_samples(database)
        self.has_seg = has_seg
        self.has_label = has_label
        self.has_info = has_info

    def __len__(self):
        return len(self.sids)

    def get_samples(self, database):
        return []

    def img_file(self, sid):
        return sid

    def seg_file(self, sid):
        return sid


class Subset(OurDataset):
    def __init__(self, dataset: OurDataset, ids) -> None:
        self.dataset = dataset
        self.ids = ids
        self.has_seg = dataset.has_seg
        self.has_label = dataset.has_label
        self.has_info = dataset.has_info
        self.sids = [dataset.sids[i] for i in ids]

    def __len__(self):
        return len(self.sids)

    def img_file(self, sid):
        return self.dataset.img_file(sid)

    def seg_file(self, sid):
        return self.dataset.seg_file(sid)


def train_subjects(x):
    sid, img, seg = x
    return Subject(
        name=sid,
        img=ScalarImage(img),
        seg=LabelMap(seg)
    )


def test_subjects(x):
    sid, img, seg = x
    return Subject(
        name=sid,
        img=ScalarImage(img),
        seg=seg,
    )


class DatasetWapper(Dataset):

    def __init__(self, dataset: OurDataset, transforms: Compose = None) -> None:
        super().__init__()
        self.ds = dataset
        self.ts = transforms

    def __getitem__(self, index):
        sid = self.ds.sids[index]
        if self.ds.has_seg:
            img = self.ds.img_file(sid)
            seg = self.ds.seg_file(sid)
            subject = Subject(
                name=sid,
                img=ScalarImage(img),
                seg=LabelMap(seg)
            )
        else:
            subject = Subject(
                name=sid,
                img=ScalarImage(self.ds.img_file(sid))
            )
        if self.ts:
            return self.ts(subject)
        else:
            subject.load()
            return subject

    def __len__(self):
        return len(self.ds)

    def pipeline(self):
        preprocess = []
        for each in self.ts.transforms:
            if isinstance(each, RandomTransform):
                break
            preprocess.append(each)
        process = []
        flag = True
        for each in self.ts.transforms:
            if flag and not isinstance(each, RandomTransform):
                flag = False
                continue
            process.append(each)
        preprocess = Compose(preprocess)
        process = Compose(process)
        pipe = iter.IterableWrapper(self.ds.sids)
        pipe = pipe.sharding_filter()
        pipe_sids, pipe_imgs, pipe_segs = pipe.fork(num_instances=3)
        pipe_imgs = iter.Mapper(pipe_imgs, self.ds.img_file)
        pipe_segs = iter.Mapper(pipe_segs, self.ds.seg_file)
        pipe = iter.Zipper(pipe_sids, pipe_imgs, pipe_segs)
        if self.ds.has_seg:
            pipe = iter.Mapper(pipe, train_subjects)
        else:
            pipe = iter.Mapper(pipe, test_subjects)
        pipe = iter.Mapper(pipe, preprocess)
        # pipe = iter.InMemoryCacheHolder(pipe) # torchData Version >= 0.4.1
        pipe = iter.Mapper(pipe, process)
        return iter.Shuffler(pipe)

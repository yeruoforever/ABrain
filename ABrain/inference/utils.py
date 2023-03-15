from typing import *
import torch
import torch.utils.data as data
from torchio import Transform, Subject, Pad, DATA


class UNet3DGridPatch(data.Dataset):
    '''生成用于3D U-Net推断的图像patch块。

    ### Args:
    - `subject:     Torchio.Subject`
    - `patch_size:  Tuple[int]`
        用于预测的patch块的大小
    - `output_size: Tuple[int]`
        模型产生的预测概率图大小
    - `Transforms:  Optional[Torchio.Transform]`

    ### Note:
        * `Transforms`中无需添加padding， GridPath会根据`patch_size`和`output_size`自动填补padding。

    ### Example:
    ```python
    >>> dataset = UNet3DGridPatch(subject, [128, 128, 128], [36, 36, 36])
    >>> aggregator = UNet3DGridAggregator(subject)
    >>> for img, locations in data.DataLoader(dataset, batch_size=args.batch_size):
            logits = model(img.to(device))
            aggregator.add_batch(logits, locations)
    >>> output = aggregator.get_output_tensor()
    >>> torch.save(output.cpu(), name)
    >>> label = output.argmax(dim=0)
    ```
    '''

    def fixed_bounds(self, width, size, step):
        ans = []
        s = 0
        while True:
            ans.append(s)
            s += step
            if s+width > size:
                ans.append(size-width)
                break
        return torch.tensor(ans).unique()

    def __init__(self, subject: Subject, patch_size: Tuple[int], output_size: Tuple[int], transforms: Optional[Transform] = None) -> None:
        super().__init__()
        self.subject = subject if transforms is None else transforms(subject)
        w, h, d = output_size
        x, y, z = patch_size
        patch_size = torch.tensor(patch_size, dtype=int)
        output_size = torch.tensor(output_size, dtype=int)
        padding = (patch_size-output_size).div(2,
                                               rounding_mode='trunc').tolist()
        padding = Pad(padding=padding, padding_mode='reflect')
        self.subject = padding(self.subject)
        W, H, D = self.subject.shape[-3:]
        X = self.fixed_bounds(x, W, w)
        Y = self.fixed_bounds(y, H, h)
        Z = self.fixed_bounds(z, D, d)
        starts = torch.cartesian_prod(X, Y, Z)
        ends = starts + patch_size
        ends_output = starts + output_size
        self.bounds = torch.cat((starts, ends), dim=1)
        self.locations = torch.cat((starts, ends_output), dim=1)

    def __getitem__(self, index):
        a, b, c, A, B, C = self.bounds[index]
        img = self.subject['img'][DATA][:, a:A, b:B, c:C]
        location = self.locations[index]
        return img, location

    def __len__(self):
        return len(self.bounds)


class UNet3DGridAggregator(object):
    '''基于patch块来构建完整的分割概率图

    ### Args:
    - `subject:torchio.Subject`
        用于预测的输入图像
    '''

    def __init__(self, subject: Subject) -> None:
        self.output_size = subject.shape[-3:]
        self.data = None
        self.cnt = None

    def add_batch(self, patch, location) -> None:
        '''收录patch， 用于拼接完整的概率图
        ### Args:
        - `patch:Tensor`
        - `location:Tensor`
        '''
        B, C, W, H, D = patch.shape
        if self.data is None:
            self.data = torch.zeros(
                C, *self.output_size,
                dtype=patch.dtype,
                device=patch.device
            )
            self.cnt = torch.zeros(
                C, *self.output_size,
                dtype=patch.dtype,
                device=patch.device
            )
        for i in range(B):
            x, y, z, X, Y, Z = location[i]
            self.data[:, x:X, y:Y, z:Z] += patch[i]
            self.cnt[:, x:X, y:Y, z:Z] += 1.

    def get_output_tensor(self) -> torch.Tensor:
        '''返回拼接好的概率图
        '''
        res = self.data/self.cnt
        return res.cpu()

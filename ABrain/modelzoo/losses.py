import torch.nn.functional as F
from torch import Tensor, tensor
from torch.nn import Module


class DiceLoss3D(Module):
    def __init__(
        self, weight: Tensor = None, smooth: float = 1.0, ignore_bg: bool = False
    ) -> None:
        super().__init__()
        self.weight = weight
        self.smooth = smooth
        self.ignore_bg = ignore_bg

    def make_onehot(self, input: Tensor, target: Tensor):
        n_labels = input.shape[1]
        target = F.one_hot(target, n_labels)
        target = target.permute(0, 4, 1, 2, 3)  # B,W,H,D,C -> B,C,W,H,D
        return target

    def dice(self, input: Tensor, target: Tensor):
        smooth = self.smooth
        numerator = (2 * input * target).sum(dim=(2, 3, 4))
        denominator = (input * input + target).sum(dim=(2, 3, 4)) + smooth
        dice = numerator / denominator
        return dice

    def forward(self, input: Tensor, target: Tensor):
        # input size (B,C,W,H,D)
        # target size (B,W,H,D)
        target = self.make_onehot(input, target)

        if self.ignore_bg:
            input = input[:, 1:, ...]
            target = target[:, 1:, ...]
        loss = 1 - self.dice(input, target)

        if self.weight is not None:
            weight = tensor(self.weight, dtype=float, device=loss.device)
            weight.div(weight.sum())
            loss = loss * weight
        loss = loss.mean(dim=1)
        return loss.mean()


class DiceLoss2D(DiceLoss3D):
    def __init__(
        self, weight: Tensor = None, smooth: float = 1, ignore_bg: bool = False
    ) -> None:
        super().__init__(weight, smooth, ignore_bg)

    def dice(self, input, target):
        smooth = self.smooth
        numerator = (2 * input * target).sum(dim=(2, 3))
        denominator = (input * input + target).sum(dim=(2, 3)) + smooth
        dice = numerator / denominator
        return dice

    def make_onehot(self, input: Tensor, target: Tensor):
        n_labels = input.shape[1]
        target = F.one_hot(target, n_labels)
        target = target.permute(0, 3, 1, 2)  # B,W,H,C -> B,C,W,H
        return target


class GeneralizedDiceLoss3D(Module):
    """DOI: 10.1007/978-3-319-67558-9 28"""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, input: Tensor, target: Tensor):
        # input size (B,C,W,H,D)
        # target size (B,W,H,D)
        n_labels = input.shape[1]
        target = F.one_hot(target, n_labels)
        target = target.permute(0, 4, 1, 2, 3)  # B,W,H,D,C -> B,C,W,H,D
        smooth = self.smooth
        # input=input[:,1:,...]
        # target=target[:,1:,...]
        numerator = (2 * input * target).sum(dim=(1, 2, 3, 4)) + smooth
        denominator = (input + target).sum(dim=(1, 2, 3, 4)) + smooth
        dice = numerator / denominator
        loss = 1 - dice
        return loss.mean()

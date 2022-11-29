import torch.nn.functional as F
from torch import Tensor, tensor
from torch.nn import Module


class DiceLoss3D(Module):
    def __init__(self, weight: Tensor = None, smooth: float = 1., ignore_bg: bool = False) -> None:
        super().__init__()
        self.weight = weight
        self.smooth = smooth
        self.ignore_bg = ignore_bg

    def forward(self, input: Tensor, target: Tensor):
        # input size (B,C,W,H,D)
        # target size (B,W,H,D)
        n_labels = input.shape[1]
        target = F.one_hot(target, n_labels)
        target = target.permute(0, 4, 1, 2, 3)  # B,W,H,D,C -> B,C,W,H,D
        smooth = self.smooth
        if self.ignore_bg:
            input = input[:, 1:, ...]
            target = target[:, 1:, ...]
        numerator = (2*input*target).sum(dim=(2, 3, 4))
        denominator = (input*input+target).sum(dim=(2, 3, 4))+smooth
        dice = numerator/denominator
        loss = 1-dice
        if self.weight is not None:
            weight = tensor(self.weight, dtype=float, device=loss.device)
            weight.div(weight.sum())
            loss = loss*weight
        loss = loss.mean(dim=1)
        return loss.mean()


class GeneralizedDiceLoss3D(Module):
    '''DOI: 10.1007/978-3-319-67558-9 28'''

    def __init__(self, smooth: float = 1.) -> None:
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
        numerator = (2*input*target).sum(dim=(1, 2, 3, 4))+smooth
        denominator = (input*+target).sum(dim=(1, 2, 3, 4))+smooth
        dice = numerator/denominator
        loss = 1-dice
        return loss.mean()

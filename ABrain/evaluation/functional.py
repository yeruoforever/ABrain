import torch
import torch.nn.functional as F
from monai.metrics.hausdorff_distance import \
    compute_hausdorff_distance as hausdorff
from monai.metrics.surface_distance import \
    compute_average_surface_distance as asd
from torch import Tensor


def one_hot_safety(label, n_labels):
    label[label >= n_labels] = 0
    return F.one_hot(label, n_labels)


def iou(pred: Tensor, target: Tensor, n_labels: int, bg_ignore: bool = False, smooth: float = 1.):
    # pred   (B,W,H,D)
    # target (B,W,H,D)
    st = 1 if bg_ignore else 0
    ious = []
    for label in range(st, n_labels):
        R = target == label
        P = pred == label
        numerator = torch.logical_and(R, P).sum()
        denominator = torch.logical_or(P, R).sum()
        score = (numerator+smooth)/(denominator+smooth)
        ious.append(score)
    return Tensor(ious).tolist()


def iou_on_confusion_matrix(mat: Tensor, bg_ignore: bool = False, smooth: float = 1.):
    n_labels = mat.shape[0]
    st = 1 if bg_ignore else 0
    ious = []
    for label in range(st, n_labels):
        numerator = mat[label, label]
        denominator = mat[label, :].sum()+mat[:, label].sum()-numerator
        score = (numerator+smooth)/(denominator+smooth)
        ious.append(score)
    return Tensor(ious).tolist()


def assd(pred: Tensor, target: Tensor, n_labels: int, bg_ignore: bool = False):
    '''average symmetric surface distance'''
    # pred   (B,W,H,D)
    # target (B,W,H,D)
    pred = one_hot_safety(pred, n_labels)
    pred = pred.permute(0, 4, 1, 2, 3)      # B,W,H,D,C -> B,C,W,H,D
    target = one_hot_safety(target, n_labels)
    target = target.permute(0, 4, 1, 2, 3)  # B,W,H,D,C -> B,C,W,H,D
    use_bg = not bg_ignore
    res = asd(pred, target, use_bg, symmetric=True)
    return res.tolist()


def hausdorff_distance(pred: Tensor, target: Tensor, n_labels: int, bg_ignore: bool = False):
    '''hausdorff_distance'''
    # pred   (B,W,H,D)
    # target (B,W,H,D)
    pred = one_hot_safety(pred, n_labels)
    pred = pred.permute(0, 4, 1, 2, 3)      # B,W,H,D,C -> B,C,W,H,D
    target = one_hot_safety(target, n_labels)
    target = target.permute(0, 4, 1, 2, 3)  # B,W,H,D,C -> B,C,W,H,D
    use_bg = not bg_ignore
    res = hausdorff(pred, target, use_bg, percentile=95)
    return res.tolist()


def dice(pred: Tensor, target: Tensor, n_labels: int, bg_ignore: bool = False, smooth: float = 1.):
    # pred   (B,W,H,D)
    # target (B,W,H,D)
    st = 1 if bg_ignore else 0
    dices = []
    for label in range(st, n_labels):
        R = target == label
        P = pred == label
        numerator = torch.logical_and(R, P).sum()
        denominator = R.sum()+P.sum()
        dice = (2*numerator+smooth)/(denominator+smooth)
        dices.append(dice)
    return Tensor(dices).tolist()


def dice_on_confusion_matrix(mat: Tensor, bg_ignore: bool = False, smooth: float = 1.):
    n_labels = mat.shape[0]
    st = 1 if bg_ignore else 0
    dices = []
    for label in range(st, n_labels):
        denominator = mat[label, :].sum()+mat[:, label].sum()
        numerator = mat[label, label]
        score = (2*numerator+smooth)/(denominator+smooth)
        dices.append(score)
    return Tensor(dices).tolist()


def f1score(pred: Tensor, target: Tensor, n_labels: int, bg_ignore: bool = False, smooth: float = 1.):
    # pred   (B,W,H,D)
    # target (B,W,H,D)
    st = 1 if bg_ignore else 0
    f1s = []
    for label in range(st, n_labels):
        R = target == label
        P = pred == label
        TP = torch.logical_and(R, P).sum()
        # `P ^ `R = `(P U R)
        TN = torch.logical_not(torch.logical_or(P, R)).sum()
        FP = torch.logical_and(P, torch.logical_not(R)).sum()
        FN = R.numel()-TP-TN-FP
        recall = TP/(TP+FN)
        precision = TP/(TP+FP)
        f1 = 2*recall*precision/(precision+recall)
        f1s.append(f1)
    return Tensor(f1s).tolist()


def f1_on_confusion_matrix(mat: Tensor, bg_ignore: bool = False, smooth: float = 1.):
    n_labels = mat.shape[0]
    st = 1 if bg_ignore else 0
    f1s = []
    total = mat.sum()
    for label in range(st, n_labels):
        P = mat[:, label].sum()
        # N = total-P
        TP = mat[label, label]
        FP = P - TP
        FN = mat[label, :].sum()-TP
        # TN = total - TP -FP -FN
        recall = TP/(TP+FN)
        precision = TP/(TP+FP)
        score = 2*recall*precision/(precision+recall)
        f1s.append(score)
    return Tensor(f1s).tolist()


def vd(pred: Tensor, target: Tensor, n_labels: int, bg_ignore: bool = False):
    ''' Volume Difference'''
    # pred   (B,W,H,D)
    # target (B,W,H,D)
    st = 1 if bg_ignore else 0
    vds = []
    for label in range(st, n_labels):
        R = target == label
        P = pred == label
        vd = R.sum()-P.sum()
        vds.append(vd)
    return Tensor(vds).tolist()


def vd_on_confusion_matrix(mat: Tensor, bg_ignore: bool = False, smooth: float = 1.):
    n_labels = mat.shape[0]
    st = 1 if bg_ignore else 0
    vds = []
    for label in range(st, n_labels):
        vd = mat[label, :].sum()-mat[:label].sum()
        vds.append(vd)
    return Tensor(vds).tolist()


def rvd(pred: Tensor, target: Tensor, n_labels: int, bg_ignore: bool = False):
    '''Relative Volume Difference'''
    # pred   (B,W,H,D)
    # target (B,W,H,D)
    st = 1 if bg_ignore else 0
    vds = []
    for label in range(st, n_labels):
        R = target == label
        P = pred == label
        p = P.sum()
        r = R.sum()
        r = 1 if r == 0 else r
        vd = abs(p-r)/r
        vds.append(vd)
    return Tensor(vds).tolist()


def rvd_on_confusion_matrix(mat: Tensor, bg_ignore: bool = False, smooth: float = 1.):
    n_labels = mat.shape[0]
    st = 1 if bg_ignore else 0
    vds = []
    for label in range(st, n_labels):
        P = mat[:, label].sum()
        R = mat[label, :].sum()
        vd = (P-R).abs().div(R)
        vds.append(vd)
    return Tensor(vds).tolist()


def voe(pred: Tensor, target: Tensor, n_labels: int, bg_ignore: bool = False, smooth: float = 1.):
    '''volumetric overlap error'''
    # pred   (B,W,H,D)
    # target (B,W,H,D)
    st = 1 if bg_ignore else 0
    voes = []
    for label in range(st, n_labels):
        R = target == label
        P = pred == label
        numerator = torch.logical_and(R, P).sum()
        denominator = torch.logical_or(P, R).sum()
        voe = 1-(numerator+smooth)/(denominator+smooth)
        voes.append(voe)
    return Tensor(voes).tolist()


def voe_on_confusion_matrix(mat: Tensor, bg_ignore: bool = False, smooth: float = 1.):
    n_labels = mat.shape[0]
    st = 1 if bg_ignore else 0
    voes = []
    for label in range(st, n_labels):
        R = mat[label, :].sum()
        P = mat[:, label].sum()
        numerator = mat[label, label].sum()
        denominator = P + R - numerator
        voe = 1-(numerator+smooth)/(denominator+smooth)
        voes.append(voe)
    return Tensor(voes).tolist()

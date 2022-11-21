import torch


class RGBPCA(torch.nn.Module):
    '''AlexNet PCA augmentation'''

    def __init__(self) -> None:
        super().__init__()

    def forward(self, img: torch.Tensor):
        with torch.no_grad():
            C, W, H = img.shape
            rgb = img.flatten(start_dim=1)
            rgb = rgb.sub_(torch.mean(rgb, dim=1, keepdim=True))
            cov = torch.cov(rgb)
            l, u = torch.linalg.eigh(cov)
            a = torch.randn(3)*0.1
            rgb.add_(torch.matmul(u, a*l).view(3, 1))
            return rgb.reshape(C, W, H)

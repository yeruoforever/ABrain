import torch
import torch.nn as nn

from ..base import ClassifyModel


class LinearPool(nn.Module):
    '''LeNet式Pooling层'''
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.pool_w = torch.ones(in_channels, 1, 2, 2)
        self.pool_b = torch.zeros(in_channels)
        self.w = nn.parameter.Parameter(torch.randn(in_channels, 1, 1))
        self.b = nn.parameter.Parameter(torch.randn(in_channels, 1, 1))
        self.g = in_channels
        self.activate = nn.Sigmoid()

    def forward(self, x):  # （B,C,W,H）
        x = nn.functional.conv2d(x, self.pool_w, self.pool_b, 2, groups=self.g)
        x = self.w*x+self.b
        self.activate(x)
        return x

class ScaledHyperholicTanh(nn.Module):
    '''f(X)=a*Tanh(s*X)'''
    def __init__(self,a=1.7159,s=2/3) -> None:
        super().__init__()
        self.a=a
        self.s=s

    def forward(self,x):
        y = self.a*torch.tanh(self.s*x)
        return y

    

class LeNet5(ClassifyModel):
    '''A simple implementation for paper "Gradient based learning applied to document recognition."
    ### Args:

    - `n_class:int`
        the number of out labels.
    - `in_channels:int`
        the number of input channels.
    - `args`
        other positional parameters.
    - `kwargs`
        other keywords parameters.

    ### Example:
    ```python
    >>> model = LeNet5(n_class=10, in_channels=1)
    >>> x = torch.rand(8, 1, 32, 32)
    >>> y = model(x)
    >>> print(y.shape)
    >>> torch.size([8, 10])
    ```
    '''
    def __init__(self, n_class: int, in_channels: int, *args, **kwargs) -> None:
        super().__init__(n_class)
        self.c1 = nn.Conv2d(in_channels, 6, kernel_size=(5, 5))
        # 6 @ 28x28
        self.s2 = LinearPool(6)     
        # 6 @ 14x14
        self.c3 = nn.Conv2d(6, 16, kernel_size=(5, 5))         
        # 16 @ 10x10
        self.s4 = LinearPool(16)
        # 16 @ 5x5
        self.c5 = nn.Sequential(
            nn.Conv2d(16, 120, kernel_size=(5, 5)),
            nn.Flatten()
        ) 
        # 120
        self.f6 = nn.Sequential(
            nn.Linear(120,84),
            ScaledHyperholicTanh()
        )
        # 84
        
        # TODO 论文中使用的是Gaussian Connections
        self.classify=nn.Sequential(
            nn.Linear(84,self.n_class),
            nn.Softmax(-1)
        )
        # n_class

    def forward(self,x):
        x=self.c1(x)
        x=self.s2(x)
        x=self.c3(x)
        x=self.s4(x)
        x=self.c5(x)
        x=self.f6(x)
        x=self.classify(x)
        return x

    
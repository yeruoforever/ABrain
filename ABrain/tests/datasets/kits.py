import unittest

from ...dataset.Kits import Kits21
from ...dataset.base import DatasetWapper

class TestKits(unittest.TestCase):
    def setUp(self) -> None:
        pass
    
    def test_kits19(self):
        ds = Kits21()
        ds = DatasetWapper(ds,lambda x:x)
        E = 0
        SZ = []
        SP = []
        for each in ds:
            _,x,y,z=each.shape
            d,w,h=each.spacing
            SZ.append([x,y,z])
            SP.append([d,w,h])
        import numpy as np
        import matplotlib.pyplot as plt
        size = np.array(SZ)
        space = np.array(SP)
        size = size**2
        mi=space.min(axis=0)
        mx=space.max(axis=0)
        print(mi,mx)
        # x
        ans = []
        X = []
        x = mi[0]
        while x<=mx[0]:
            E = size[:,0]*(1-space[:,0]/x)**2
            E = E.sum()
            ans.append(E)
            X.append(x)
            x += 0.05
        ids=np.argmin(ans)
        print(X[ids],ans[ids])
        plt.plot(X,ans)
        plt.savefig("resample_X.jpg")
        # y
        ans = []
        X = []
        x = mi[1]
        while x<=mx[1]:
            E = size[:,1]*(1-space[:,1]/x)**2
            E = E.sum()
            ans.append(E)
            X.append(x)
            x += 0.05
        ids=np.argmin(ans)
        print(X[ids],ans[ids])
        plt.cla()
        plt.plot(X,ans)
        plt.savefig("resample_Y.jpg")
        # z
        ans = []
        X = []
        x = mi[2]
        while x<=mx[2]:
            E = size[:,2]*(1-space[:,2]/x)**2
            E = E.sum()
            ans.append(E)
            X.append(x)
            x += 0.05
        ids=np.argmin(ans)
        print(X[ids],ans[ids])
        plt.cla()
        plt.plot(X,ans)
        plt.savefig("resample_Z.jpg")

            
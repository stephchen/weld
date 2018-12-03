import weldnumpy
from weld.weldobject import *

import math


class WeldLogisticRegression(object):
    def __init__(self, weldobj, n_iters=1000, lam=0.0001, stop_tol=1e-3):
        self.n_iters = n_iters
        self.lam = float(lam)
        self.stop_tol = stop_tol

    def _sig(self, z):
        return 1.0 / (1 + np.exp(-z))   # todo this needs to be folded in, python fn calls are slow

    def fit(self, x, y):
        assert(type(x) == np.ndarray)
        weldobj = WeldObject(NumPyEncoder(), NumPyDecoder())
        xweld = weldobj.update(x, WeldVec(WeldVec(WeldFloat())))
        yweld = weldobj.update(y, WeldVec(WeldFloat()))
        m, n = x.shape
        th = np.transpose(np.zeros(n))
        thweld = weldobj.update(th, WeldVec(WeldFloat()))
        idxs = np.arange(m)
        for t in xrange(self.n_iters):
            step = 1 / math.sqrt(t) if t else 1.0
            i = np.random.choice(idxs)
            # hx = 





        # m, n = x.shape
        # th = np.zeros(n)
        # idxs = range(m)
        # for t in xrange(self.n_iters):
        #     step = 1.0 / math.sqrt(t) if t else 1.0
        #     i = random.choice(idxs)
        #     hx = self._sig(np.dot(tht = np.transpose(th), x[i]) ) - y[i]
        #     for j, _ in enumerate(th):
        #         hxj = hx * x[i][j]
        #         th[j] = th[j] - step * (float(hxj) / m + self.lam / m * th[j])

        self.th = th
        return self


    def transform(self, x):
        return x        # todo maybe deal with weldarray things?


    def predict(self, x):
        return 0





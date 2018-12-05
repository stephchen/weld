import weldnumpy
from grizzly.encoders import NumPyEncoder, NumPyDecoder
from weld.weldobject import *
import util

import math
import numpy as np


class WeldLogisticRegression(object):
    def __init__(self, n_iters=1000, lam=0.0001, stop_tol=1e-3):
        self.n_iters = n_iters
        self.lam = float(lam)
        self.stop_tol = stop_tol

    def _sig(self, z):
        return 1.0 / (1 + np.exp(-z))   # todo this needs to be folded in, python fn calls are slow

    def fit(self, x, y):        # todo x is required to be a matrix here (i think this is ok, just have brittle types)
        assert(type(x) == np.ndarray)
        assert(type(y) == np.ndarray)

        m, n = x.shape
        th = np.transpose(np.zeros(n))
        idxs = np.arange(m)

        weldobj = WeldObject(NumPyEncoder(), NumPyDecoder())

        # pregenerate idxs
        isamps = np.random.choice(idxs, self.n_iters, replace=True)
        template = """
          @(loopsize: %(niters)sL)
          iterate(
            {%(isamps)s, 0, %(th)s},
            |p| { {
              let i = lookup(p.$0, p.$1);
              let xi = lookup(%(x)s, i);
              let step = if(p.$1 > 0, 1 / sqrt(p.$1), 1.0);
              let hx = 1 / (1 + exp(0 - result(
                @(loopsize: %(th_len)sL)
                for(
                  zip(p.$2, xi),
                  merger[f32, +],
                  |b, i, e| merge(b, e.$0 * e.$1)
                  )
                ))) - lookup(%(y)s, i);
              map(
                p.$2, |th| th - step * (result(
                  map(zip(p.$2, xi), |p2| p2.$0 - step * (hx * p2.$1 / %(m)s + %(lam)s / %(m)s * p2.$0))
                ))
              )
            }, p.$1 < %(niters)s }.$2
          )"""

        weldobj.weld_code = template % {
            'niters': weldobj.update(self.n_iters, WeldDouble()),
            'isamps': weldobj.update(isamps, WeldVec(WeldDouble())),
            'th':weldobj.update(th, WeldVec(WeldDouble())),
            'th_len': weldobj.update(len(th), WeldDouble()),
            'x': weldobj.update(x, WeldVec(WeldVec(WeldDouble()))),
            'y': weldobj.update(y, WeldVec(WeldDouble())),
            'm': weldobj.update(m, WeldDouble()),
            'lam': weldobj.update(self.lam, WeldDouble())
        }

        import pdb
        pdb.set_trace()

        self.th = weldobj.evaluate(WeldVec(WeldDouble()))

        return self


    def transform(self, x):
        return x


    def predict(self, x):
        # we require x to be a vector (ie single sample) here, since optimizing for single inference requests (todo batch?)
        ret_ = self._sig(util.dot_vv(self.th, x))
        return 1.0 if ret_ >= 0.5 else 0.0


    def score(self, x, y):
        # todo for this to be useful we need batch predict, otherwise just clientside check is fine
        pass





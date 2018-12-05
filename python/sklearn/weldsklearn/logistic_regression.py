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

    def fit(self, x, y, weldobj=None):        # todo x is required to be a matrix here (i think this is ok, just have brittle types)
        self.weldobj = weldobj if weldobj else WeldObject(NumPyEncoder(), NumPyDecoder())

        m, n = x.shape
        th = np.zeros(n, dtype=np.float32)
        idxs = np.arange(m, dtype=np.int64)

        # pregenerate idxs
        isamps = np.random.choice(idxs, self.n_iters, replace=True)

        template = """
          @(loopsize: %(niters)sL)
          iterate(
            {%(isamps)s, i64(0), %(th)s},
            |p| { {
              p.$0, p.$1 + i64(1),
                let i = lookup(p.$0, p.$1);
                let xi = lookup(%(x)s, i);
                let step = if(p.$1 > i64(0), f32(1) / sqrt(f32(p.$1)), f32(1));
                let hx = f32(1) / (f32(1) + exp(f32(0) - f32(result(
                  @(loopsize: %(th_len)sL)
                  for(
                    zip(p.$2, xi),
                    merger[f32, +],
                    |b, ii, e| merge(b, e.$0 * e.$1)
                  )
                )))) - f32(lookup(%(y)s, i));

                result(@(loopsize: %(th_len)sL)
                  for(
                    p.$2, appender[f32], |b, j, e| merge(b, e - f32(step) * (f32(hx) * lookup(xi, j) / f32(%(m)s) + f32(%(lam)s) / f32(%(m)s) * e))
                  ))
            }, p.$1 < i64(%(niters)s) }).$2"""


        self.weldobj.weld_code = template % {
            'niters': str(self.n_iters),
            'isamps': weldobj.update(isamps, WeldVec(WeldLong())),
            'th':weldobj.update(th, WeldVec(WeldFloat())),
            'th_len': str(len(th)),
            'x': weldobj.update(x, WeldVec(WeldVec(WeldFloat()))),
            'y': weldobj.update(y, WeldVec(WeldFloat())),
            'm': str(float(m)),
            'lam': str(float(self.lam))
        }
        self.th = self.weldobj.evaluate(WeldVec(WeldFloat()))

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





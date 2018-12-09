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

    def fit(self, x, y):        # todo x is required to be a matrix here (i think this is ok, just have brittle types)
        assert(type(x) == np.ndarray)
        assert(type(y) == np.ndarray)

        m, n = x.shape
        th = np.zeros(n, dtype=np.float32)
        idxs = np.arange(m, dtype=np.int64)

        weldobj = WeldObject(NumPyEncoder(), NumPyDecoder())

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


        weldobj.weld_code = template % {
            'niters': str(self.n_iters),
            'isamps': weldobj.update(isamps, WeldVec(WeldLong())),
            'th':weldobj.update(th, WeldVec(WeldFloat())),
            'th_len': str(len(th)),
            'x': weldobj.update(x, WeldVec(WeldVec(WeldFloat()))),
            'y': weldobj.update(y, WeldVec(WeldFloat())),
            'm': str(float(m)),
            'lam': str(float(self.lam))
        }
        self.th = weldobj.evaluate(WeldVec(WeldFloat()))

        return self


    def transform(self, x):
        return x


    def predict(self, x):
        template = """
            f32(1) / (f32(1) + exp(f32(0) - result(
                @(loopsize: %(th_len)sL)
                for(
                    zip(%(th)s, %(x)s),
                    merger[f32, +],
                    |b, i, e| merge(b, e.$0 * e.$1)
                )
            )))
        """
        weldobj = WeldObject(NumPyEncoder(), NumPyDecoder())
        weldobj.weld_code = template % {
            'th': weldobj.update(self.th, WeldVec(WeldFloat())),
            'x': weldobj.update(x, WeldVec(WeldFloat())),
            'th_len': str(len(self.th))
        }

        ret_ = weldobj.evaluate(WeldFloat(), verbose=False)
        return 1.0 if ret_ >= 0.5 else 0.0


    def score(self, x, y):
        template = """
            @(loopsize: %(xlen)sL)
            result(for(
                zip(%(x)s, %(y)s),
                merger[f32, +],
                |b, i, e| let res = f32(1) / (f32(1) + exp(f32(0) - result(
                    @(loopsize: %(thlen)sL)
                    for(
                        zip(%(th)s, e.$0),
                        merger[f32, +],
                        |b2, i2, e2| merge(b2, e2.$0 * e2.$1)
                    )
                )));
                if(res >= f32(0.5) && e.$1 == f32(1.0), merge(b, f32(1)), merge(b, f32(0)))
            ))
        """

        weldobj = WeldObject(NumPyEncoder(), NumPyDecoder())
        weldobj.weld_code = template % {
            'th': weldobj.update(self.th, WeldVec(WeldFloat())),
            'x': weldobj.update(x, WeldVec(WeldVec(WeldFloat()))),
            'y': weldobj.update(y, WeldVec(WeldFloat())),
            'xlen': str(len(x)),
            'thlen': str(len(self.th))
        }

        score = weldobj.evaluate(WeldVec(WeldFloat()))

        return score / len(x)








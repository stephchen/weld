import weldnumpy
from grizzly.encoders import NumPyEncoder, NumPyDecoder
from weld.weldobject import *
import util

import math
import numpy as np


class WeldKNeighbors(object):

    def fit(self, x, y):
        self.xtrain = x
        self.ytrain = y

    def transform(self, x, y):
        return x


    def predict(self, x):
        # single sample
        weldobj = WeldObject(NumPyEncoder(), NumPyDecoder())


        
        template = """
            let idx = 
            @(loopsize: %(xlen)sL)
            iterate(
                { %(xtrain)s, 0L, f32(1000000), 0L},
                |p|
                    let xt = lookup(p.$0, p.$1);
                    let dist = sqrt(result(
                        for(zip(%(x)s, xt), merger[f32, +], |b, i, e| merge(b, (e.$0 - e.$1) * (e.$0 - e.$1)))
                    ));
                    let mindist = select(dist < p.$2, dist, p.$2);
                    let minidx = select(dist < p.$2, p.$1, p.$3);

                    { {
                        p.$0, p.$1 + 1L, 
                        mindist, minidx
                    }, p.$1 < %(xlen)sL - 1L}).$3;
            lookup(%(ytrain)s, idx)
        """

        distinit = np.ndarray((1,), dtype=np.float32)
        distinit[0] = 10000000

        weldobj.weld_code = template % {
            'xtrain': weldobj.update(self.xtrain, WeldVec(WeldVec(WeldFloat()))),
            'ytrain': weldobj.update(self.ytrain, WeldVec(WeldFloat())),
            'x': weldobj.update(x, WeldVec(WeldFloat())),
            'xlen': str(len(self.xtrain)),
            'distinit': weldobj.update(distinit, WeldVec(WeldFloat()))
        }
        res = weldobj.evaluate(WeldFloat())
        # res = weldobj.evaluate(WeldFloat())

        return res

    def score(self, x, y):
        score = 0
        for i, xt in enumerate(x):
            pred = self.predict(xt)
            print i, pred, y[i]
            if pred == y[i]:
                score += 1
        return float(score) / len(x)



class NaiveKNeighbors(object):
    def fit(self, x, y):
        self.xtrain = x
        self.ytrain = y
        return self


    def transform(self, x, y):
        return x

    def predict(self, x):
        closest = None
        mindist = float('inf')
        for i, xt in enumerate(self.xtrain):
            dist = np.linalg.norm(x - xt)
            if dist < mindist:
                mindist = dist
                closest = i
        return self.ytrain[closest]


    def score(self, x, y):
        score = 0
        for i, xt in enumerate(x):
            pred = self.predict(xt)
            print i, pred, y[i]
            if pred == y[i]:
                score += 1
            if i % 100 == 0: print i
        return float(score) / len(x)




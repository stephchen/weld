import numpy as np

from grizzly.encoders import *
import weldnumpy
from weld.weldobject import *


class WeldStandardScaler(object):
    def __init__(self):
        # precompile? need to string all together and evaluate at end
        pass


    def fit(self, x, y=None, weldobj=None):
        self.weldobj = weldobj if weldobj else WeldObject(NumPyEncoder(), NumPyDecoder())

        if isinstance(x, WeldObject):
            self.fit_x = x.obj_id
            self.weldobj.dependencies[self.fit_x] = x

            mean_template = """result(for(%(fit_x)s, merger[f32, +], |b, i, e| merge(b, e))) / f32(len(%(fit_x)s))"""
            stdev_template = """
                let mean = f32(%(mean)s);
                sqrt(result(for(%(fit_x)s, merger[f32, +], |b, i, e| merge(b, (e - mean) * (e - mean)))) / f32(len(%(fit_x)s)))
            """
            self.mean = WeldObject(NumPyEncoder(), NumPyDecoder())
            self.mean.update(x)
            self.mean.dependencies[x.obj_id] = x
            self.mean.weld_code = mean_template % {'fit_x': x.obj_id}

            self.std = WeldObject(NumPyEncoder(), NumPyDecoder())
            self.std.update(self.mean)
            self.std.update(x)
            self.std.dependencies[self.mean.obj_id] = self.mean
            self.std.dependencies[x.obj_id] = x
            self.std.weld_code = stdev_template % {
                'mean': self.mean.obj_id,
                'fit_x': x.obj_id
            }
        else:
            self.mean = np.mean(x, axis=0, keepdims=True, dtype=np.float32)[0]
            self.std = np.std(x, axis=0, keepdims=True, dtype=np.float32)[0]

        return self

    def transform(self, x):
        xweld = self.weldobj.update(x, WeldVec(WeldVec(WeldFloat())))
        if isinstance(x, WeldObject):
            xweld = x.obj_id
            self.weldobj.dependencies[xweld] = x

        meanweld = self.weldobj.update(self.mean)       # mean and std should contain dependencies on old x
        if isinstance(self.mean, WeldObject):
            meanweld = self.mean.obj_id
            self.weldobj.dependencies[meanweld] = self.mean

        stdweld = self.weldobj.update(self.std)
        if isinstance(self.std, WeldObject):
            stdweld = self.std.obj_id
            self.weldobj.dependencies[stdweld] = self.std

        template = """map(%(xweld)s, |e|
            map(
                zip(map(zip(e, %(meanweld)s), |p| p.$0 - p.$1), %(stdweld)s
                ), |q| if(q.$1 == 0.0f, q.$0, q.$0 / q.$1)
            )
        )"""

        self.weldobj.weld_code = template % {
            'xweld': xweld,
            'meanweld': meanweld,
            'stdweld': stdweld
        }
        return self.weldobj.evaluate(WeldVec(WeldVec(WeldFloat())))          # todo return just the weldvec






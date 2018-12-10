import numpy as np

from grizzly.encoders import *
import weldnumpy
from weld.weldobject import *


# TODO build out base
class WeldStandardScaler(object):
    def __init__(self, weldobj=None):
        # precompile? need to string all together and evaluate at end
        pass


    def fit(self, x, y=None):
        # y is ignored
        assert(type(x) == np.ndarray)

        self.mean = np.mean(x, axis=0, keepdims=True, dtype=np.float32)
        self.std = np.std(x, axis=0, keepdims=True, dtype=np.float32)
        return self

    def transform(self, x):
        assert(type(x) == np.ndarray)
        weldobj = WeldObject(NumPyEncoder(), NumPyDecoder())
        xweld = weldobj.update(x, WeldVec(WeldVec(WeldFloat())))

        meanweld = weldobj.update(self.mean[0], WeldVec(WeldFloat()))

        stdweld = weldobj.update(self.std[0], WeldVec(WeldFloat()))

        template = """map(%(xweld)s, |e|
            map(
                zip(map(zip(e, %(meanweld)s), |p| p.$0 - p.$1), %(stdweld)s
                ), |q| if(q.$1 == 0.0f, q.$0, q.$0 / q.$1)
            )
        )"""

        weldobj.weld_code = template % {
            'xweld': xweld,
            'meanweld': meanweld,
            'stdweld': stdweld
        }
        return weldobj.evaluate(WeldVec(WeldVec(WeldFloat())))          # todo return just the weldvec



class WeldOneHotEncoder(object):
    def fit(self, x, y=None):
        # y ignored


        return self

    def transform(self, x):
        pass


class WeldBinarizer(object):
    pass
    # set feature values to 0 or 1 acc to threshold





import numpy as np

# from weld.encoders import NumpyArrayEncoder, NumpyArrayDecoder
from grizzly.encoders import NumPyEncoder, NumPyDecoder
import weldnumpy
from weld.weldobject import *


# TODO build out base
class WeldStandardScaler(object):

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

        template = """map({0}, |e|
            map(
                zip(map(zip(e, {1}), |p| p.$0 - p.$1), {2}
                ), |q| if(q.$1 == 0.0f, q.$0, q.$0 / q.$1)
            )
        )"""

        weldobj.weld_code = template.format(xweld, meanweld, stdweld)
        return weldobj.evaluate(WeldVec(WeldVec(WeldFloat())))


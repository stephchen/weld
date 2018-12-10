import numpy as np
from weld.weldobject import *
from grizzly.encoders import NumPyEncoder, NumPyDecoder



class WeldPipeline(object):
    def __init__(self, steps):
        self.classifier = steps[-1]
        self.transformers = steps[:-1]
        self.weldobj = WeldObject(NumPyEncoder(), NumPyDecoder())


    def fit(self, x, y):
        xt = x
        for tf in self.transformers:
            tf.fit(xt, y, weldobj=self.weldobj)
            xt = tf.transform(xt)
        self.classifier.fit(xt, y, weldobj=self.weldobj)
        return self


    def transform(self, x):
        xt = x
        for tf in self.transformers:
            tf.fit(xt, y, weldobj=self.weldobj)
            xt = tf.transform(xt, weldobj=self.weldobj)
        if len(x.shape) == 1:
            return self.weldobj.evaluate(WeldVec(WeldFloat()))
        elif len(x.shape) == 2:
            return self.weldobj.evaluate(WeldVec(WeldVec(WeldFloat())))


    def predict(self, x):
        xt = self.transform(x, weldobj=self.weldobj)
        return self.classifier.predict(xt, weldobj=self.weldobj)


    def score(self, x, y):
        xt = self.transform(x, weldobj=self.weldobj)
        return self.classifier.score(xt, y, weldobj=self.weldobj)
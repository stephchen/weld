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
        pass


    def predict(self, x):
        pass


    def score(self, x, y):
        pass
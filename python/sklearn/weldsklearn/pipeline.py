import numpy as np
from weld.weldobject import *
from grizzly.encoders import NumPyEncoder, NumPyDecoder



# class WeldPipeline(object):
#     def __init__(self, steps):
#         self.classifier = steps[-1]
#         self.transformers = steps[:-1]
#         self.weldobj = WeldObject(NumPyEncoder(), NumPyDecoder())


#     def fit(self, x, y):
#         xt = x
#         for tf in self.transformers:
#             tf.fit(xt, y, weldobj=self.weldobj)
#             xt = tf.transform(xt)
#         self.classifier.fit(xt, y, weldobj=self.weldobj)
#         return self


#     def transform(self, x):
#         xt = x
#         for tf in self.transformers:
#             tf.fit(xt, y, weldobj=self.weldobj)
#             xt = tf.transform(xt, weldobj=self.weldobj)
#         if len(x.shape) == 1:
#             return self.weldobj.evaluate(WeldVec(WeldFloat()))
#         elif len(x.shape) == 2:
#             return self.weldobj.evaluate(WeldVec(WeldVec(WeldFloat())))


#     def predict(self, x):
#         xt = self.transform(x, weldobj=self.weldobj)
#         return self.classifier.predict(xt, weldobj=self.weldobj)


#     def score(self, x, y):
#         xt = self.transform(x, weldobj=self.weldobj)
#         return self.classifier.score(xt, y, weldobj=self.weldobj)


class WeldPipeline(object):
    def __init__(self, steps):
        if not steps:
            raise Exception('needs at least 1 step')
        self.classifier = steps[-1]
        self.transformers = []
        if len(steps) > 1:
            for trans in steps[:-1]:
                # handle final step separately even if it is also a transformer
                self.transformers.append(trans)

    def fit(self, x, y):
        xt = x
        for i in xrange(len(self.transformers)):
            self.transformers[i].fit(xt, x)
            xt = self.transformers[i].transform(xt)
        self.classifier.fit(xt, y)
        return self

    def transform(self, x, feedout=False):
        xt = x
        for i in xrange(len(self.transformers)):
            xt = self.transformers[i].transform(xt)
        if hasattr(self.classifier, 'transform') and feedout:
            # make it an explicit option to include classifier transform, kinda assuming
            # we're not going to predict on it but that shouldn't really matter
            xt = self.classifier.transform(xt)
        return xt

    def predict(self, x):
        xt = self.transform(x)  # feedout = False
        return self.classifier.predict(xt)

    def score(self, x, y):
        correct = 0
        xt = self.transform(x)  # feedout = False
        preds = np.transpose(self.classifier.predict(xt))
        import pdb
        pdb.set_trace()
        m, _ = xt.shape
        for i in xrange(m):
            if preds[i] == y[i]:
                correct += 1
        return float(correct) / m

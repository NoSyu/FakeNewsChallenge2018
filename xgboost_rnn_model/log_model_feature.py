from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class LogModelFeature(BaseEstimator, TransformerMixin):

    def __init__(self, xgb, gru):
        self.xgb = xgb
        self.gru = gru

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.hstack((self.xgb.transform(X), self.gru.transform(X)))

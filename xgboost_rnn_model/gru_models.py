from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class GRULime(BaseEstimator, TransformerMixin):

    def __init__(self, model):

        self.model = model

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.model.predict(np.array(X))

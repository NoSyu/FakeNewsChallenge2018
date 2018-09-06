from sklearn.base import BaseEstimator, TransformerMixin


class XGBoostLime(BaseEstimator, TransformerMixin):

    def __init__(self, model):

        self.model = model

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.model.predict(X)

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class LogAmountTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        self._validate_input(X)

        X["log_amount"] = np.log1p(X["Amount"])

        return X

    def _validate_input(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        if "Amount" not in X.columns:
            raise ValueError("Input data must contain an 'Amount' column.")
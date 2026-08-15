import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class TransactionTypeEncoder(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.categories_ = np.sort(X["type"].unique())
        return self

    def transform(self, X):
        X = X.copy()

        self._validate_input_type(X)
        self._validate_type_column(X)

        X = self._encode_transaction_type(X)

        return X

    def _validate_input_type(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

    def _validate_type_column(self, X):
        if "type" not in X.columns:
            raise ValueError("Input data must contain a 'type' column.")

        unseen_categories = set(X["type"].unique()) - set(self.categories_)

        if unseen_categories:
            raise ValueError(
                f"Unseen transaction types found: {unseen_categories}"
            )

    def _encode_transaction_type(self, X):
        for category in self.categories_:
            X[f"type_{category}"] = (
                X["type"] == category
            ).astype(np.int8)

        X = X.drop(columns=["type"])

        return X
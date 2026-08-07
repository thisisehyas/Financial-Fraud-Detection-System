from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class DataCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, required_columns=None):
        self.required_columns = required_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        self._validate_input_type(X)

        X = X.copy()
        
        self._validate_required_columns(X)
        self._validate_missing_values(X)

        return X

    def _validate_input_type(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "X must be a pandas DataFrame."
            )
    
    def _validate_required_columns(self, X):
        missing_columns = set(self.required_columns) - set(X.columns)

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {sorted(missing_columns)}"
            )

    def _validate_missing_values(self, X):
        missing_counts = X.isna().sum()
        missing_columns = missing_counts[missing_counts > 0]

        if not missing_columns.empty:
            raise ValueError(
                f"Missing values detected:\n{missing_columns}"
            )

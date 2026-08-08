from sklearn.base import BaseEstimator, TransformerMixin

class TransactionHistoryPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X = X.sort_values(
            ["nameOrig", "step"]
        ).reset_index(drop=True)

        return X
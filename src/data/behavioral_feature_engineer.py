from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class BehavioralFeatureEngineer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        return X

    def _create_transaction_count_features(self, X):
        groups = X.groupby("nameOrig").indices
        steps_all = X["step"].values

        windows = {
            "24h": 24,
            "72h": 72,
            "7d": 168,
        }

        for name, window in windows.items():
            tx_count = np.zeros(len(X), dtype=np.int32)

            for idx in groups.values():
                steps = steps_all[idx]
                j = 0

                for i in range(len(steps)):
                    while steps[j] < steps[i] - window:
                        j += 1
                    tx_count[idx[i]] = i - j

            X[f"tx_count_last_{name}"] = tx_count

        X["tx_count_total_user"] = (
            X
            .groupby("nameOrig")["step"]
            .transform("count")
        )

        return X

    def _create_amount_features(self, X):
        groups = X.groupby("nameOrig").indices
        steps_all = X["step"].values
        amounts_all = X["amount"].values

        windows = {
            "24h": 24,
            "7d": 168,
        }

        for name, window in windows.items():
            sum_amount = np.zeros(len(X), dtype=np.float32)

            for idx in groups.values():
                steps = steps_all[idx]
                amounts = amounts_all[idx]

                j = 0
                running_sum = 0.0

                for i in range(len(steps)):
                    while steps[j] < steps[i] - window:
                        running_sum -= amounts[j]
                        j += 1

                    sum_amount[idx[i]] = running_sum

                    running_sum += amounts[i]

            X[f"sum_amount_last_{name}"] = sum_amount

            count = X[f"tx_count_last_{name}"]

            X[f"avg_amount_last_{name}"] = np.divide(
                sum_amount,
                count,
                out=np.zeros_like(sum_amount),
                where=count != 0,
            )

        return X

    def _create_destination_features(self, X):
        groups = X.groupby("nameOrig").indices
        steps_all = X["step"].values

        dest_codes, _ = X["nameDest"].factorize()
        dest_codes = dest_codes.astype(np.int32)

        windows = {
            "24h": 24,
            "72h": 72,
            "7d": 168,
        }

        for name, window in windows.items():
            result = np.zeros(len(X), dtype=np.int32)

            for idx in groups.values():
                steps = steps_all[idx]

                start = 0

                for i in range(len(idx)):
                    while steps[start] < steps[i] - window:
                        start += 1

                    seen = set()

                    for k in range(start, i):
                        seen.add(dest_codes[idx[k]])

                    result[idx[i]] = len(seen)

            X[f"unique_dest_last_{name}"] = result

        return X
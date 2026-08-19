from sklearn.pipeline import Pipeline

from log_amount_transformer import LogAmountTransformer


creditcard_pipeline = Pipeline([
    ("log_amount", LogAmountTransformer()),
])
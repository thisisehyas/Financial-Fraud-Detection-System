from sklearn.pipeline import Pipeline

from src.data.log_amount_transformer import LogAmountTransformer


def create_creditcard_pipeline():
    return Pipeline([
        ("log_amount", LogAmountTransformer()),
    ])
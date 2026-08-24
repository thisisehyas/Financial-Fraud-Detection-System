from sklearn.pipeline import Pipeline
from sklearn import set_config

from src.data.cleaners import DataCleaner
from src.data.transaction_history_preprocessor import TransactionHistoryPreprocessor
from src.data.behavioral_feature_engineer import BehavioralFeatureEngineer
from src.data.transaction_type_encoder import TransactionTypeEncoder
from src.data.feature_scaler import scaler


set_config(transform_output="pandas")


def create_paysim_pipeline():
    return Pipeline([
        (
            "cleaner",
            DataCleaner(
                required_columns=[
                    "step",
                    "type",
                    "amount",
                    "nameOrig",
                    "nameDest",
                ]
            ),
        ),

        ("history", TransactionHistoryPreprocessor()),

        ("behavioral_features", BehavioralFeatureEngineer()),

        ("type_encoder", TransactionTypeEncoder()),

        ("scaler", scaler),
    ])
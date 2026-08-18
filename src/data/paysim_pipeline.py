from sklearn.pipeline import Pipeline
from sklearn import set_config

from cleaners import DataCleaner
from transaction_history_preprocessor import TransactionHistoryPreprocessor
from behavioral_feature_engineer import BehavioralFeatureEngineer
from transaction_type_encoder import TransactionTypeEncoder
from feature_scaler import scaler

set_config(transform_output="pandas")

paysim_pipeline = Pipeline([
    ("cleaner", DataCleaner(required_columns=[
        "step",
        "type",
        "amount",
        "nameOrig",
        "nameDest",
    ])),

    ("history", TransactionHistoryPreprocessor()),

    ("behavioral_features", BehavioralFeatureEngineer()),

    ("type_encoder", TransactionTypeEncoder()),

    ("scaler", scaler),

])

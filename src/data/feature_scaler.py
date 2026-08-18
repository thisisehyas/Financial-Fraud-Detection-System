from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

numerical_features = [
    "step",
    "amount",

    # Transaction count features
    "tx_count_last_24h",
    "tx_count_last_72h",
    "tx_count_last_7d",
    "tx_count_total_user",

    # Amount features
    "sum_amount_last_24h",
    "avg_amount_last_24h",
    "sum_amount_last_7d",
    "avg_amount_last_7d",

    # Destination features
    "unique_dest_last_24h",
    "unique_dest_last_72h",
    "unique_dest_last_7d",
]

scaler = ColumnTransformer(
    transformers=[
        (
            "numerical",
            StandardScaler(),
            numerical_features
        )
    ],
    remainder="passthrough"
)
import pandas as pd
from sklearn.model_selection import train_test_split


def split_paysim(
    X,
    test_size=0.2,
    random_state=42,
):
    """
    Split PaySim data at the user level.

    All transactions belonging to the same origin account
    are kept in the same split.
    """

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    required_columns = {"nameOrig", "isFraud"}

    missing_columns = required_columns - set(X.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Create one label per user
    user_labels = (
        X.groupby("nameOrig")["isFraud"]
        .max()
        .reset_index()
    )

    train_users, test_users = train_test_split(
        user_labels,
        test_size=test_size,
        random_state=random_state,
        stratify=user_labels["isFraud"],
    )

    train_user_ids = set(train_users["nameOrig"])
    test_user_ids = set(test_users["nameOrig"])

    X_train = X[X["nameOrig"].isin(train_user_ids)].copy()
    X_test = X[X["nameOrig"].isin(test_user_ids)].copy()

    return X_train, X_test
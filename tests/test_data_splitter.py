from src.data.data_splitter import split_paysim
import pandas as pd

paysim = pd.read_csv("data/raw/paysim.csv")

# Use a small subset for testing
test_data = paysim.sample(
    n=100_000,
    random_state=42
).reset_index(drop=True)


# Run the split
train, test = split_paysim(
    test_data,
    test_size=0.2,
    random_state=42
)


print("Train shape:", train.shape)
print("Test shape:", test.shape)

train_users = set(train["nameOrig"])
test_users = set(test["nameOrig"])

overlap = train_users & test_users

assert len(overlap) == 0, (
    f"Data leakage detected! "
    f"{len(overlap)} users appear in both train and test."
)

print("✓ No user appears in both train and test.")

assert len(train) + len(test) == len(test_data)

assert (
    len(
        pd.concat([train, test])
        .drop_duplicates()
    )
    == len(test_data)
)

print("✓ No transactions were lost or duplicated.")

original_users = set(test_data["nameOrig"])
split_users = train_users | test_users

assert original_users == split_users

print("✓ All users are preserved.")

train_user_labels = (
    train.groupby("nameOrig")["isFraud"]
    .max()
)

test_user_labels = (
    test.groupby("nameOrig")["isFraud"]
    .max()
)

train_fraud_rate = train_user_labels.mean()
test_fraud_rate = test_user_labels.mean()

print("Train fraud-user rate:", train_fraud_rate)
print("Test fraud-user rate:", test_fraud_rate)

assert abs(train_fraud_rate - test_fraud_rate) < 0.02

print("✓ Fraud-user ratio is approximately preserved.")

print("Train users:", train["nameOrig"].nunique())
print("Test users:", test["nameOrig"].nunique())
print("Total users:", test_data["nameOrig"].nunique())

assert (
    train["nameOrig"].nunique()
    + test["nameOrig"].nunique()
    == test_data["nameOrig"].nunique()
)

print("✓ Users were split correctly.")

train_2, test_2 = split_paysim(
    test_data,
    test_size=0.2,
    random_state=42
)

assert train["nameOrig"].equals(train_2["nameOrig"])
assert test["nameOrig"].equals(test_2["nameOrig"])

print("✓ Split is reproducible.")
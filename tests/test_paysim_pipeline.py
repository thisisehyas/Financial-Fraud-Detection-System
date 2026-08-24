import numpy as np
import pandas as pd

from src.data.paysim_pipeline import create_paysim_pipeline


train = pd.read_csv("data/splits/paysim_train.csv")
test = pd.read_csv("data/splits/paysim_test.csv")

print("Train:", train.shape)
print("Test:", test.shape)


# create a fresh pipeline
paysim_pipeline = create_paysim_pipeline()

# fit only on training data
paysim_pipeline.fit(train)

print("✓ PaySim pipeline fitted successfully.")


# Transform both datasets
train_processed = paysim_pipeline.transform(train)
test_processed = paysim_pipeline.transform(test)

print("Processed train shape:", train_processed.shape)
print("Processed test shape:", test_processed.shape)


# --------------------------------------------------
# 1. Train and test must have the same feature count
# --------------------------------------------------

assert train_processed.shape[1] == test_processed.shape[1]

print("✓ Train and test have the same number of features.")


# --------------------------------------------------
# 2. Number of output rows must be preserved
# --------------------------------------------------

assert train_processed.shape[0] == train.shape[0]
assert test_processed.shape[0] == test.shape[0]

print("✓ No transactions were lost during transformation.")


# --------------------------------------------------
# 3. Output must not contain NaN or infinite values
# --------------------------------------------------

assert np.isfinite(train_processed.to_numpy()).all()
assert np.isfinite(test_processed.to_numpy()).all()

print("✓ No NaN or infinite values in processed data.")


# --------------------------------------------------
# 4. Expected output feature count
# --------------------------------------------------

expected_feature_count = 24

assert train_processed.shape[1] == expected_feature_count
assert test_processed.shape[1] == expected_feature_count

print("✓ Expected number of output features is correct.")


print("\nAll PaySim pipeline tests passed successfully.")
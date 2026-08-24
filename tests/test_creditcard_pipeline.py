import numpy as np
import pandas as pd

from src.data.creditcard_pipeline import create_creditcard_pipeline


creditcard = pd.read_csv(
    "data/raw/creditcard.csv"
)

print("Credit Card shape:", creditcard.shape)


# Create a fresh pipeline
pipeline = create_creditcard_pipeline()


# --------------------------------------------------
# 1. Fit pipeline
# --------------------------------------------------

pipeline.fit(creditcard)

print("✓ Credit Card pipeline fitted successfully.")


# --------------------------------------------------
# 2. Transform data
# --------------------------------------------------

processed = pipeline.transform(creditcard)

print("Processed shape:", processed.shape)


# --------------------------------------------------
# 3. Number of rows must be preserved
# --------------------------------------------------

assert processed.shape[0] == creditcard.shape[0]

print("✓ No rows were lost during transformation.")


# --------------------------------------------------
# 4. log_amount must exist
# --------------------------------------------------

assert "log_amount" in processed.columns

print("✓ log_amount feature was created.")


# --------------------------------------------------
# 5. Check log transformation
# --------------------------------------------------

expected_log_amount = np.log1p(
    creditcard["Amount"]
)

assert np.allclose(
    processed["log_amount"],
    expected_log_amount
)

print("✓ log_amount was calculated correctly.")


# --------------------------------------------------
# 6. No NaN or infinite values
# --------------------------------------------------

assert np.isfinite(
    processed["log_amount"].to_numpy()
).all()

print("✓ No NaN or infinite values in log_amount.")


print("\nAll Credit Card pipeline tests passed successfully.")
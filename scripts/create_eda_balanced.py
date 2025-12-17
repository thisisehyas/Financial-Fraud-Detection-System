import pandas as pd
from pathlib import Path

RAW_DATA_PATH = Path("data/raw/paysim.csv")
OUTPUT_PATH = Path("data/eda/paysim_balanced_eda.csv")

print("Loading raw data...")
paysim = pd.read_csv(RAW_DATA_PATH)

fraud_df = paysim[paysim['isFraud'] == 1]
non_fraud_df = paysim[paysim['isFraud'] == 0]

print(f"Total fraud transactions: {len(fraud_df)}")
print(f"Total non-fraud transactions: {len(non_fraud_df)}")

N_SAMPLES = 1000
RANDOM_STATE = 42

fraud_sample = fraud_df.sample(
    n=N_SAMPLES,
    random_state=RANDOM_STATE
)

non_fraud_sample = non_fraud_df.sample(
    n=N_SAMPLES,
    random_state=RANDOM_STATE    
)

balanced_eda_df = pd.concat(
    [fraud_sample, non_fraud_sample],
    axis=0
).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
balanced_eda_df.to_csv(OUTPUT_PATH, index=False)

print(f"balanced EDA dataset saved to: {OUTPUT_PATH}")
print(balanced_eda_df["isFraud"].value_counts())
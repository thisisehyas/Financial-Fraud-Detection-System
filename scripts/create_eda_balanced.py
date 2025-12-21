import pandas as pd
from pathlib import Path

# config
RAW_DATA_PATH = Path("data/raw/creditcard.csv")
OUTPUT_PATH = Path("data/eda/creditcard_balanced_eda.csv")
LABEL_COL = 'Class'

MAX_SAMPLES = 1000
RANDOM_STATE = 42

# load data
print(f"Loading raw data from {RAW_DATA_PATH}...")
df = pd.read_csv(RAW_DATA_PATH)

# split classes
fraud_df = df[df[LABEL_COL] == 1]
non_fraud_df = df[df[LABEL_COL] == 0]

N_SAMPLES = min(MAX_SAMPLES, len(fraud_df))

print(f"Total fraud transactions: {len(fraud_df)}")
print(f"Total non-fraud transactions: {len(non_fraud_df)}")

# sample
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

# save
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
balanced_eda_df.to_csv(OUTPUT_PATH, index=False)

print(f"balanced EDA dataset saved to: {OUTPUT_PATH}")
print(balanced_eda_df[LABEL_COL].value_counts())
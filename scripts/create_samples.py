"""
Create stratified samples for:
    - data/raw/creditcard.csv  -> data/sample/creditcard_sample.csv
    - data/raw/paysim.csv      -> data/sample/paysim_sample.csv

Usage:
    python scritpts/create_samples.py
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

RAW_DIR = "data/raw"
SAMPLE_DIR = "data/sample"
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)

KAGGLE_FILE = os.path.join(RAW_DIR, "creditcard.csv")
PAYSIM_FILE = os.path.join(RAW_DIR, "paysim.csv")

KAGGLE_SAMPLE_OUT = os.path.join(SAMPLE_DIR, "creditcard_sample.csv")
PAYSIM_SAMPLE_OUT = os.path.join(SAMPLE_DIR, "paysim_sample.csv")

SAMPLE_SIZE = 1000
RANDOM_SEED = 42
CHUNKSIZE = 200_000
MAX_FRAUD_MEM = 200_000

np.random.seed(RANDOM_SEED)

def stratified_sample_kaggle(in_path=KAGGLE_FILE, out_path=KAGGLE_SAMPLE_OUT, sample_size=SAMPLE_SIZE):
    print("== Kaggle credicard sampling ==")
    if not os.path.exists(in_path):
        print(f"Error: {in_path} not found.")
        return
    df = pd.read_csv(in_path)
    n = min(sample_size, len(df))
    if "Class" not in df.columns:
        print("Warning: 'Class' column not found - doing simple random sample")
        sample_df = df.sample(n=n, random_state=RANDOM_SEED)
    else:
        # if one of the classes has fewer samples than required stratified split would fail
        counts = df["Class"].value_counts()
        if (counts < 1).any():
            sample_df = df.sample(n=n, random_state=RANDOM_SEED)
        else:
            try:
                sample_df, _ = train_test_split(df, train_size=n, stratify=df["Class"], random_state=RANDOM_SEED)
            except Exception as e:
                # fallback to group-wise sampling
                print("Stratified split failed:", e)
                # simple group-wise sampling preserving class ration approximately
                proportions = df["Class"].value_counts(normalize=True).to_dict()
                picks = []
                for cls, prop in proportions.items():
                    k = max(1, int(round(prop * n)))
                    cls_rows = df[df["Class"] == cls]
                    k = min(k, len(cls_rows)) # if not enough rows, reduce k
                    picks.append(cls_rows.sample(n=k, random_state=RANDOM_SEED))
                sample_df = pd.contcat(picks).sample(n=n, random_state=RANDOM_SEED) # final shuffle
    
    sample_df.to_csv(out_path, index=False)
    print("Saved Kaggle sample to:", out_path)
    if "Class" in sample_df.columns:
        print("Class distribution in sample:\n", sample_df["Class"].value_counts())

def count_paysim(in_path=PAYSIM_FILE, chunksize=CHUNKSIZE):
    total = 0
    fraud = 0
    print("Counting PaySim rows and frauds...")

    try:
        for chunk in pd.read_csv(in_path, chunksize=chunksize):
            total += len(chunk)
            if "isFraud" in chunk.columns:
                fraud += int(chunk["isFraud"].sum())
            else:
                raise ValueError("PaySim file does not contain 'isFraud' column.")
    except pd.errors.EmptyDataError:
        raise

    print(f"Total rows: {total}, Total fraud rows: {fraud}")
    return total, fraud

def stratified_sample_paysim(in_path=PAYSIM_FILE, out_path=PAYSIM_SAMPLE_OUT, sample_size=SAMPLE_SIZE, max_fraud_mem=MAX_FRAUD_MEM, chunksize=CHUNKSIZE):
    print("== PaySim sampling ==")
    if not os.path.exists(in_path):
        print(f"Error: {in_path} not found.")
        return
    
    total, total_fraud = count_paysim(in_path, chunksize=chunksize)

    if total == 0:
        print("Empty PaySim file. Exiting.")
        return
    
    prop_fraud = total_fraud / total
    n_fraud_desired = int(round(prop_fraud * sample_size))

    # ensure at least 1 fraud if there is any
    if total_fraud > 0 and n_fraud_desired == 0:
        n_fraud_desired = 1
    
    n_fraud_desired = min(n_fraud_desired, sample_size - 1)
    n_nonfraud_desired = sample_size - n_fraud_desired

    print(f"Dataset fraud proportion: {prop_fraud:.6f}")
    print(f"Target sample size: {sample_size} (Fraud: {n_fraud_desired}, Non-fraud: {n_nonfraud_desired})")


    # if total fraud is small enough, collect all fraud rows
    collect_all_fraud = (total_fraud <= max_fraud_mem)

    fraud_rows = []
    nonfraud_reservoir = []
    nonfraud_seen = 0

    if collect_all_fraud:
        print("Collecting all fraud rows into memory.")
        # first pass: collect fraud and reservoir-sample nonfraud
        reader = pd.read_csv(in_path, chunksize=chunksize)

        for i, chunk in enumerate(reader):
            fraud_chunk = chunk[chunk["isFraud"] == 1]
            if not fraud_chunk.empty:
                fraud_rows.append(fraud_chunk)

            nonfraud_chunk = chunk[chunk["isFraud"] == 0]
            for _, row in nonfraud_chunk.iterrows():
                nonfraud_seen += 1  
                if len(nonfraud_reservoir) < n_nonfraud_desired:
                    nonfraud_reservoir.append(row)
                else:
                    r = np.random.randint(0, nonfraud_seen)
                    if r < n_nonfraud_desired:
                        nonfraud_reservoir[r] = row
            print(f"Processed chunk {i+1} - fraud collected so far: {sum(len(f) for f in fraud_rows)}, nonfraud reservoir: {len(nonfraud_reservoir)}", end="\r")

        fraud_df = pd.concat(fraud_rows, ignore_index=True) if fraud_rows else pd.DataFrame(columns=chunk.columns)

    else:
        # If fraud is large we will reservoir-sample for both fraud and non-fraud
        print("Using reservoir sampling for both fraud and non-fraud (fraud large).")
        fraud_reservoir = []
        fraud_seen = 0
        reader = pd.read_csv(in_path, chunksize=chunksize)
        for i, chunk in enumerate(reader):
            # fraud chunk reservoir sampling
            fchunk = chunk[chunk["isFraud"] == 1]
            for _, row in fchunk.iterrows():
                fraud_seen += 1
                if len(fraud_reservoir) < n_fraud_desired:
                    fraud_reservoir.append(row)
                else:
                    r = np.random.randint(0, fraud_seen)
                    if r < n_fraud_desired:
                        fraud_reservoir[r] = row
            # non-fraud reservoir
            nfchunk = chunk[chunk["isFraud"] == 0]
            for _, row in nfchunk.iterrows():
                nonfraud_seen += 1
                if len(nonfraud_reservoir) < n_nonfraud_desired:
                    nonfraud_reservoir.append(row)
                else:
                    r = np.random.randint(0, nonfraud_seen)
                    if r < n_nonfraud_desired:
                        nonfraud_reservoir[r] = row
            print(f"Processed chunk {i+1} - fraud_seen: {fraud_seen}, nonfraud_seen: {nonfraud_seen}", end="\r")
        fraud_df = pd.DataFrame(fraud_reservoir)

    # decide how many frauds we can actually use
    n_fraud_available = len(fraud_df)
    n_fraud_to_use = min(n_fraud_desired, n_fraud_available)

    # sample frauds (if more availabe than needed)
    if n_fraud_available > 0:
        fraud_sample = fraud_df.sample(n=n_fraud_to_use, random_state=RANDOM_SEED)
    else:
        fraud_sample = pd.DataFrame(columns=fraud_df.columns)

    # sample non-frauds from reservoir
    nonfraud_df = pd.DataFrame(nonfraud_reservoir)
    n_non_needed = sample_size - len(fraud_sample)
    if len(nonfraud_df) == 0 and n_non_needed > 0:
        raise RuntimeError("No non-fraud rows collected, something went wrong.")
    nonfraud_sample = nonfraud_df.sample(n=min(n_non_needed, len(nonfraud_df)), random_state=RANDOM_SEED)

    sample_df = pd.concat([fraud_sample, nonfraud_sample], ignore_index=True)
    # If we have fewer than sample_size (e.g., not enough nonfraud), attempt to fill by sampling more nonfraud if possible
    if len(sample_df) < sample_size:
        short = sample_size - len(sample_df)
        print(f"\nWarning: only {len(sample_df)} rows collected, trying to add {short} more non-fraud from reservoir.")
        extra = nonfraud_df.sample(n=min(short, max(0, len(nonfraud_df)-len(nonfraud_sample))), random_state=RANDOM_SEED)
        sample_df = pd.concat([sample_df, extra], ignore_index=True)

    # final shuffle
    sample_df = sample_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # save
    sample_df.to_csv(out_path, index=False)
    print(f"\nSaved PaySim sample to: {out_path}")
    if "isFraud" in sample_df.columns:
        print("isFraud distribution in sample:\n", sample_df["isFraud"].value_counts())
    print("Sample shape:", sample_df.shape)
    print()


def main():
    try:
        stratified_sample_kaggle()
    except Exception as e:
        print("Kaggle sampling failed:", e, file=sys.stderr)

    try:
        stratified_sample_paysim()
    except Exception as e:
        print("PaySim sampling failed:", e, file=sys.stderr)


if __name__ == "__main__":
    main()
    
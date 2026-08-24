import os
import pandas as pd

from src.data.data_splitter import split_paysim


RAW_PATH = "data/raw/paysim.csv"
TRAIN_PATH = "data/splits/paysim_train.csv"
TEST_PATH = "data/splits/paysim_test.csv"


os.makedirs("data/splits", exist_ok=True)

paysim = pd.read_csv(RAW_PATH)

train, test = split_paysim(
    paysim,
    test_size=0.2,
    random_state=42,
)

train.to_csv(TRAIN_PATH, index=False)
test.to_csv(TEST_PATH, index=False)

print(f"Train saved: {TRAIN_PATH} → {train.shape}")
print(f"Test saved: {TEST_PATH} → {test.shape}")
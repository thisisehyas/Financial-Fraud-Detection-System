import joblib
import pandas as pd

from src.data.paysim_pipeline import create_paysim_pipeline
from src.data.creditcard_pipeline import create_creditcard_pipeline

paysim_train = pd.read_csv("data/splits/paysim_train.csv")

creditcard = pd.read_csv("data/raw/creditcard.csv")

#PaySim Dataset
paysim_pipeline = create_paysim_pipeline()

paysim_pipeline.fit(paysim_train)

joblib.dump(
    paysim_pipeline, 
    "models/paysim_pipeline.joblib"
)

# CreditCard Dataset
creditcard_pipeline = create_creditcard_pipeline()

creditcard_pipeline.fit(creditcard)

joblib.dump(
    creditcard_pipeline,
    "models/creditcard_pipeline.joblib"
)

# Verify saved pipelines
loaded_paysim_pipeline = joblib.load(
    "models/paysim_pipeline.joblib"
)

loaded_creditcard_pipeline = joblib.load(
    "models/creditcard_pipeline.joblib"
)

paysim_sample = paysim_train.head(100)

paysim_output = loaded_paysim_pipeline.transform(
    paysim_sample
)

assert len(paysim_output) == len(paysim_sample)

print("✓ Saved PaySim pipeline loaded successfully.")


creditcard_sample = creditcard.head(100)

creditcard_output = loaded_creditcard_pipeline.transform(
    creditcard_sample
)

assert len(creditcard_output) == len(creditcard_sample)

print("✓ Saved Credit Card pipeline loaded successfully.")
# incremental_dataset.py
import os
import pandas as pd

DATASET_PATH = "datasets/click_fraud_incremental.csv"

def append_to_dataset(row_dict):
    df = pd.DataFrame([row_dict])

    os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)

    if not os.path.exists(DATASET_PATH):
        df.to_csv(DATASET_PATH, index=False)
    else:
        df.to_csv(DATASET_PATH, mode="a", header=False, index=False)

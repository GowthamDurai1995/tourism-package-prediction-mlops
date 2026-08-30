
import pandas as pd
import os

DATA_PATH = "data/tourism.csv"

def register_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    print("Data registered successfully!")
    print(f"Dataset shape: {df.shape}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isnull().sum())

if __name__ == "__main__":
    register_data()

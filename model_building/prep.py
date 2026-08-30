
import os
from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path

def prepare_data():
    # Define paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_PATH = BASE_DIR / "data" / "tourism.csv"

    # Folder for processed data
    OUTPUT_DIR = BASE_DIR / "data" / "processed"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Remove unnecessary columns
    columns_to_drop = ["Unnamed: 0", "CustomerID"]
    df.drop(columns=columns_to_drop, inplace=True, errors="ignore")

    # Separate features and target
    X = df.drop("ProdTaken", axis=1)
    y = df["ProdTaken"]

    # Split into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save processed datasets
    X_train.to_csv(OUTPUT_DIR / "X_train.csv", index=False)
    X_test.to_csv(OUTPUT_DIR / "X_test.csv", index=False)
    y_train.to_csv(OUTPUT_DIR / "y_train.csv", index=False)
    y_test.to_csv(OUTPUT_DIR / "y_test.csv", index=False)

    print("Data preparation completed successfully!")
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")


if __name__ == "__main__":
    prepare_data()

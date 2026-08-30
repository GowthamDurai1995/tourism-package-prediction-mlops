
import os
import glob
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# PROJECT PATHS
# ============================================================

# The script is expected to run from the project root:
# /content/tourism_project in Colab
# repository root in GitHub Actions
PROJECT_ROOT = Path.cwd()

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "model_building" / "model"

# Input files
X_TRAIN_PATH = DATA_DIR / "X_train.csv"
X_TEST_PATH = DATA_DIR / "X_test.csv"
Y_TRAIN_PATH = DATA_DIR / "y_train.csv"
Y_TEST_PATH = DATA_DIR / "y_test.csv"

# Output files
MODEL_PATH = MODEL_DIR / "tourism_model.pkl"
EXPERIMENT_RESULTS_PATH = MODEL_DIR / "experiment_results.csv"


# ============================================================
# VALIDATE PROJECT PATH
# ============================================================

def validate_paths():
    """Check whether all required processed data files exist."""

    required_files = [
        X_TRAIN_PATH,
        X_TEST_PATH,
        Y_TRAIN_PATH,
        Y_TEST_PATH
    ]

    missing_files = [str(file) for file in required_files if not file.exists()]

    if missing_files:
        raise FileNotFoundError(
            "The following required files are missing:\n"
            + "\n".join(missing_files)
            + f"\n\nCurrent working directory: {PROJECT_ROOT}"
        )

    print("All processed data files found successfully.")


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """Load training and testing datasets."""

    print("\nLoading processed data...")

    X_train = pd.read_csv(X_TRAIN_PATH)
    X_test = pd.read_csv(X_TEST_PATH)

    y_train = pd.read_csv(Y_TRAIN_PATH).squeeze()
    y_test = pd.read_csv(Y_TEST_PATH).squeeze()

    print(f"Training features shape: {X_train.shape}")
    print(f"Testing features shape: {X_test.shape}")
    print(f"Training target shape: {y_train.shape}")
    print(f"Testing target shape: {y_test.shape}")

    return X_train, X_test, y_train, y_test


# ============================================================
# BUILD PIPELINE
# ============================================================

def build_pipeline(X_train):
    """Create preprocessing and Random Forest model pipeline."""

    # Identify categorical and numerical columns
    categorical_columns = X_train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numerical_columns = X_train.select_dtypes(
        exclude=["object", "category", "bool"]
    ).columns.tolist()

    print("\nCategorical columns:")
    print(categorical_columns)

    print("\nNumerical columns:")
    print(numerical_columns)

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_columns
            ),
            (
                "numerical",
                "passthrough",
                numerical_columns
            )
        ]
    )

    # Model
    model = RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    )

    # Complete pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    return pipeline


# ============================================================
# TRAIN AND TUNE MODEL
# ============================================================

def train_model():
    """Train, tune and evaluate the tourism prediction model."""

    validate_paths()

    # Load data
    X_train, X_test, y_train, y_test = load_data()

    # Build pipeline
    pipeline = build_pipeline(X_train)

    print("\nStarting model training and hyperparameter tuning...")

    # Hyperparameter combinations
    param_grid = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2]
    }

    # Grid search
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=3,
        scoring="f1",
        n_jobs=-1,
        verbose=1
    )

    # Train
    grid_search.fit(X_train, y_train)

    # Best model
    best_model = grid_search.best_estimator_

    print("\nTraining completed successfully!")

    print("\nBest Parameters:")
    print(grid_search.best_params_)

    print(
        f"\nBest Cross-Validation F1 Score: "
        f"{grid_search.best_score_:.4f}"
    )

    # ========================================================
    # EVALUATION
    # ========================================================

    predictions = best_model.predict(X_test)

    test_accuracy = accuracy_score(y_test, predictions)
    test_f1 = f1_score(y_test, predictions, zero_division=0)

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0
    )

    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    # ========================================================
    # SAVE MODEL AND RESULTS
    # ========================================================

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save trained model
    joblib.dump(best_model, MODEL_PATH)

    # Save experiment results
    results = pd.DataFrame([
        {
            "best_cv_f1_score": grid_search.best_score_,
            "test_accuracy": test_accuracy,
            "test_f1_score": test_f1,
            "best_parameters": str(grid_search.best_params_),
            "weighted_precision": report["weighted avg"]["precision"],
            "weighted_recall": report["weighted avg"]["recall"],
            "weighted_f1_score": report["weighted avg"]["f1-score"]
        }
    ])

    results.to_csv(
        EXPERIMENT_RESULTS_PATH,
        index=False
    )

    print(f"\nExperiment results saved at: {EXPERIMENT_RESULTS_PATH}")
    print(f"Best model saved at: {MODEL_PATH}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    train_model()


import os
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------
# Paths
# -------------------------------
DATA_DIR = "/content/tourism_project/data/processed"
MODEL_DIR = "/content/tourism_project/model_building/model"

os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------------
# Load training and testing data
# -------------------------------
X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
X_test = pd.read_csv(f"{DATA_DIR}/X_test.csv")

y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv").squeeze()
y_test = pd.read_csv(f"{DATA_DIR}/y_test.csv").squeeze()

# -------------------------------
# Identify column types
# -------------------------------
categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X_train.select_dtypes(exclude=["object"]).columns.tolist()

# -------------------------------
# Preprocessing pipeline
# -------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("numerical", "passthrough", numerical_cols)
    ]
)

# -------------------------------
# Base model
# -------------------------------
rf = RandomForestClassifier(
    random_state=42,
    class_weight="balanced"
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", rf)
    ]
)

# -------------------------------
# Parameters for experimentation
# -------------------------------
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2]
}

# -------------------------------
# Hyperparameter tuning
# -------------------------------
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=3,
    scoring="f1",
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# -------------------------------
# Best model
# -------------------------------
best_model = grid_search.best_estimator_

print("\nTraining completed successfully!")
print("\nBest Parameters:")
print(grid_search.best_params_)

print(f"\nBest Cross-Validation F1 Score: {grid_search.best_score_:.4f}")

# -------------------------------
# Test evaluation
# -------------------------------
y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nTest Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -------------------------------
# Log all experiments
# -------------------------------
results = pd.DataFrame(grid_search.cv_results_)

experiment_columns = [
    "params",
    "mean_test_score",
    "std_test_score",
    "rank_test_score"
]

results[experiment_columns].sort_values(
    by="rank_test_score"
).to_csv(
    f"{MODEL_DIR}/experiment_results.csv",
    index=False
)

print(f"\nExperiment results saved at: {MODEL_DIR}/experiment_results.csv")

# -------------------------------
# Save best model
# -------------------------------
model_path = f"{MODEL_DIR}/tourism_model.pkl"
joblib.dump(best_model, model_path)

print(f"Best model saved at: {model_path}")


# Student Performance Predictor
# Train and evaluate a machine-learning model

from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Step 1: Define project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "student-mat.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"

# Step 2: Load the dataset
if not DATA_PATH.exists():
    raise FileNotFoundError(
        "Dataset not found. Put student-mat.csv inside the data folder."
    )

df = pd.read_csv(DATA_PATH, sep=";")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# Step 3: Select input features and target
features = ["studytime", "failures", "absences", "G1", "G2"]
target = "G3"

X = df[features]
y = df[target]

# Step 4: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# Step 5: Create the machine-learning pipeline
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("regressor", RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=2,
        random_state=42
    ))
])

# Step 6: Train the model
print("Training the model...")
model.fit(X_train, y_train)

# Step 7: Evaluate predictions on the test data
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("\nModel Evaluation")
print("----------------")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R2 Score: {r2:.2f}")

# Step 8: Save the trained model and evaluation results
MODEL_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(model, MODEL_PATH)

metadata = {
    "features": features,
    "target": target,
    "mae": round(float(mae), 3),
    "rmse": round(float(rmse), 3),
    "r2": round(float(r2), 3),
    "test_rows": int(len(y_test))
}

with open(METADATA_PATH, "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=4)

print("\nTraining completed successfully!")
print("Saved model:", MODEL_PATH)
print("Saved metadata:", METADATA_PATH)

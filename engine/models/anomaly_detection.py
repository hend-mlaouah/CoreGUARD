from pathlib import Path
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

BASE_DIR = Path(__file__).resolve().parents[2]  
PROCESSED_DIR = BASE_DIR / "data" / "processed"   
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True, parents=True)

INPUT_PATH = PROCESSED_DIR / "dataset_features_reduites.csv"
OUTPUT_PATH = PROCESSED_DIR / "dataset_anomalies.csv"

MODEL_PATH = MODEL_DIR / "isolation_forest.pkl"

# ==========================
# PARAMETERS
# ==========================
CONTAMINATION = 0.10
N_ESTIMATORS = 300
RANDOM_STATE = 42

# ==========================
# LOAD DATA
# ==========================
print(f"[LOAD] dataset_features_reduites <- {INPUT_PATH.resolve()}")
df = pd.read_csv(INPUT_PATH, index_col=0)
df.index = pd.to_datetime(df.index)
print(f"Dataset : {df.shape}")

X = df.select_dtypes(include="number")
print(f"Variables utilisées : {len(X.columns)}")

# ==========================
# TRAIN
# ==========================
model = IsolationForest(
    contamination=CONTAMINATION,
    n_estimators=N_ESTIMATORS,
    random_state=RANDOM_STATE
)
pred = model.fit_predict(X)
score = model.decision_function(X)

joblib.dump(model, MODEL_PATH)
print(f"[SAVE] isolation_forest <- {MODEL_PATH.resolve()}")

result = df.copy()
result["anomaly"] = (pred == -1).astype(int)
result["anomaly_score"] = score

print("\nNombre d'anomalies :")
print(result["anomaly"].value_counts())
print("\nPourcentage :")
print(result["anomaly"].mean() * 100)

anomalies = result[result["anomaly"] == 1]
print("\nAnomalies par heure")
print(anomalies.index.hour.value_counts().sort_index())
print("\nAnomalies par jour")
print(anomalies.index.dayofweek.value_counts().sort_index())

result.to_csv(OUTPUT_PATH)
print(f"[SAVE] dataset_anomalies -> {OUTPUT_PATH.resolve()}")
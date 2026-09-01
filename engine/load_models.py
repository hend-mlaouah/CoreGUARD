from pathlib import Path
import pickle
import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

ISOLATION_FOREST_PATH = MODELS_DIR / "isolation_forest.pkl"
XGBOOST_PATH = MODELS_DIR / "xgboost_incident_raw.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

DATASET_ANOMALIES_PATH = DATA_DIR / "dataset_anomalies.csv"
DATASET_FEATURES_REDUITES_PATH = DATA_DIR / "dataset_features_reduites.csv"


def _load_pickled_model(path: Path):
    """Charge un modèle sauvegardé via joblib.dump — joblib d'abord,
    repli sur pickle standard si besoin."""
    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


@st.cache_resource(show_spinner="Chargement du modèle Isolation Forest...")
def load_isolation_forest():
    print(f"[LOAD] isolation_forest <- {ISOLATION_FOREST_PATH.resolve()}")
    return _load_pickled_model(ISOLATION_FOREST_PATH)


@st.cache_resource(show_spinner="Chargement du modèle XGBoost...")
def load_xgboost_model():
    print(f"[LOAD] xgboost <- {XGBOOST_PATH.resolve()}")
    return _load_pickled_model(XGBOOST_PATH)


@st.cache_resource(show_spinner="Chargement du scaler d'entraînement...")
def load_scaler():
    """Retourne le scaler sauvegardé à l'entraînement. Le scaler est
    désormais OBLIGATOIRE (plus de repli silencieux) — voir
    check_required_files()."""
    print(f"[LOAD] scaler <- {SCALER_PATH.resolve()}")
    if not SCALER_PATH.exists():
        return None
    try:
        return _load_pickled_model(SCALER_PATH)
    except Exception:
        return None


@st.cache_data(show_spinner="Chargement du dataset d'anomalies...")
def load_anomalies_dataset() -> pd.DataFrame:
    return pd.read_csv(DATASET_ANOMALIES_PATH)


@st.cache_data(show_spinner="Chargement des features réduites...")
def load_features_dataset() -> pd.DataFrame:
    return pd.read_csv(DATASET_FEATURES_REDUITES_PATH)


def check_required_files() -> list[str]:
    # scaler.pkl est maintenant obligatoire : plus de silence possible
    # sur un fallback local qui fausse les prédictions.
    required = [ISOLATION_FOREST_PATH, XGBOOST_PATH, SCALER_PATH]
    return [str(p) for p in required if not p.exists()]
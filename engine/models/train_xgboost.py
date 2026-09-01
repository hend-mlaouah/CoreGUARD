import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]   
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "dataset_labels.csv"   
MODEL_OUTPUT_PATH = PROJECT_ROOT / "models" / "xgboost_incident_raw.pkl"  
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reports"                         

LABEL_COLUMN = "label_incident"
TEST_SIZE = 0.2
RANDOM_STATE = 42

KPIS_LABEL_SOURCE = [
    "pdc__avg_login_time",
    "pdc__num-cmds-fail-timeout",
    "pdc__timeout_rate",
    "pdc__timeout-occured",
    "pgw__drop_rate_ul",
    "pgw__uplink_dropped-packets__delta",
]


def get_leakage_columns(columns, kpi_bases):
    """Retourne toutes les colonnes (brutes + dérivées roll/lag/diff)
    correspondant aux KPIs utilisés pour construire le label."""
    leaked = [c for c in columns if any(c.startswith(kpi) for kpi in kpi_bases)]
    return leaked

def compute_scale_pos_weight(y):
    n_neg = np.sum(y == 0)
    n_pos = np.sum(y == 1)
    return n_neg / max(n_pos, 1)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig") 
    df.columns = df.columns.str.strip()  

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        print("[INFO] Dataset trié chronologiquement par 'timestamp'.")
    else:
        print("[ATTENTION] Pas de colonne 'timestamp' — impossible de garantir "
              "un split chronologique correct.")

    print(f"[INFO] Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df


def prepare_features(df: pd.DataFrame, label_col: str):
    if label_col not in df.columns:
        candidates = [
            c for c in df.columns
            if any(k in c.lower() for k in ["label", "anomaly", "incident", "target", "class"])
        ]
        print(f"[ERREUR] Colonnes disponibles dans le dataset ({len(df.columns)}) :")
        print([repr(c) for c in df.columns])
        if candidates:
            print(f"[SUGGESTION] Colonnes candidates pour LABEL_COLUMN : {candidates}")
        raise ValueError(
            f"Colonne label '{label_col}' introuvable dans le dataset. "
            f"Vérifie le nom exact ci-dessus et mets à jour LABEL_COLUMN."
        )

    y = df[label_col]
    X = df.drop(columns=[label_col])

    if "timestamp" in X.columns:
        X = X.drop(columns=["timestamp"])

    leakage_cols = get_leakage_columns(X.columns, KPIS_LABEL_SOURCE)
    if leakage_cols:
        print(f"[INFO] {len(leakage_cols)} colonnes retirées (fuite avec le label) :")
        print(leakage_cols)
        X = X.drop(columns=leakage_cols)
    else:
        print("[ATTENTION] Aucune colonne de fuite détectée — vérifie KPIS_LABEL_SOURCE.")

   
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        print(f"[INFO] Encodage des colonnes catégorielles restantes : {non_numeric}")
        for col in non_numeric:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    return X, y


def train_model(X_train, y_train):
    scale_pos_weight = compute_scale_pos_weight(y_train)
    print(f"[INFO] scale_pos_weight calculé : {scale_pos_weight:.2f}")

    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    return model


def cross_validate(model, X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = cross_val_score(model, X, y, cv=tscv, scoring="f1")
    print(f"[INFO] F1 en cross-validation temporelle ({n_splits} folds) : "
          f"{scores.mean():.3f} ± {scores.std():.3f}")
    return scores


def evaluate_model(model, X_test, y_test, threshold=0.5):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    print(f"\n[RÉSULTATS] Classification report (seuil={threshold:.3f}) :")
    print(classification_report(y_test, y_pred, digits=3))

    print("[RÉSULTATS] Matrice de confusion :")
    print(confusion_matrix(y_test, y_pred))

    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    print(f"[RÉSULTATS] ROC-AUC : {auc:.3f} | F1-score (seuil={threshold:.3f}) : {f1:.3f}")

    return {"roc_auc": auc, "f1_score": f1}


def find_best_threshold(model, X_test, y_test):
    """Optionnel : ajuste le seuil de décision plutôt que 0.5 fixe."""
    y_proba = model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    print(f"[INFO] Meilleur seuil (max F1) : {best_threshold:.3f} "
          f"(F1={f1_scores[best_idx]:.3f})")
    return best_threshold


def save_feature_importance(
    model,
    feature_names,
    output_path=None
):
    if output_path is None:
        output_path = OUTPUT_DIR / "xgboost_feature_importance.csv"

    importance = model.feature_importances_

    fi_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    }).sort_values("importance", ascending=False)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fi_df.to_csv(output_path, index=False)

    print(f"[INFO] Importance des features sauvegardée : {output_path}")
    print(fi_df.head(15))

    return fi_df

def main():
    df = load_data(DATA_PATH)
    X, y = prepare_features(df, LABEL_COLUMN)
    split_idx = int(len(X) * (1 - TEST_SIZE))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    print(f"[INFO] Train : {X_train.shape[0]} lignes (plus anciennes) | "
          f"Test : {X_test.shape[0]} lignes (plus récentes)")
    print(f"[INFO] Répartition label train : {y_train.value_counts().to_dict()} | "
          f"test : {y_test.value_counts().to_dict()}")

    model = train_model(X_train, y_train)

    cross_validate(model, X_train, y_train)

    metrics_default = evaluate_model(model, X_test, y_test, threshold=0.5)
    best_threshold = find_best_threshold(model, X_test, y_test)
    print(f"\n[INFO] Ré-évaluation au seuil optimal :")
    metrics = evaluate_model(model, X_test, y_test, threshold=best_threshold)
    save_feature_importance(model, X.columns.tolist())

    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT_PATH)
    print(f"[INFO] Modèle sauvegardé : {MODEL_OUTPUT_PATH}")

    metrics_out = {
        "roc_auc": metrics["roc_auc"],
        "f1_score": metrics["f1_score"],
        "best_threshold": float(best_threshold),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
    }
    METRICS_PATH = OUTPUT_DIR / "xgboost_metrics.json"

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_out, f, indent=2)

    print(f"[INFO] Métriques sauvegardées : {METRICS_PATH}")


if __name__ == "__main__":
    main()
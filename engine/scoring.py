from __future__ import annotations
import numpy as np
import pandas as pd


def get_model_feature_names(model) -> list[str]:
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if hasattr(model, "get_booster"):
        try:
            names = model.get_booster().feature_names
            if names:
                return list(names)
        except Exception:
            pass
    return []


def score_isolation_forest(model, model_input: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    X = model_input.reindex(columns=feature_names, fill_value=0.0)
    pred = model.predict(X)
    score = model.decision_function(X)
    out = pd.DataFrame(index=model_input.index)
    out["anomaly"] = (pred == -1).astype(int)
    out["anomaly_score"] = score
    return out


def score_xgboost(model, model_input: pd.DataFrame, feature_names: list[str], threshold: float = 0.5) -> pd.DataFrame:
    X = model_input.reindex(columns=feature_names, fill_value=0.0)
    proba = model.predict_proba(X)[:, 1]
    out = pd.DataFrame(index=model_input.index)
    out["incident_proba"] = proba
    out["incident_pred"] = (proba >= threshold).astype(int)
    return out


def compute_severity(df: pd.DataFrame, score_col: str = "incident_proba") -> pd.Series:
    if score_col == "incident_proba":
        bins = [0, 0.3, 0.6, 0.85, 1.0 + 1e-9]
        labels = ["Faible", "Moyen", "Élevé", "Critique"]
        return pd.cut(df[score_col], bins=bins, labels=labels, include_lowest=True).astype(str)

    if df[score_col].nunique() <= 1:
        return pd.Series(["Faible"] * len(df), index=df.index)
    pct = df[score_col].rank(pct=True)
    bins = [0, 0.5, 0.75, 0.95, 1.0 + 1e-9]
    labels = ["Faible", "Moyen", "Élevé", "Critique"]
    return pd.cut(pct, bins=bins, labels=labels, include_lowest=True).astype(str)


def build_scored_dataset(model_input: pd.DataFrame, iso_result: pd.DataFrame, xgb_result: pd.DataFrame) -> pd.DataFrame:
    scored = model_input.join(iso_result).join(xgb_result)
    scored["severity"] = compute_severity(scored, "incident_proba")
    scored["row_id"] = np.arange(len(scored))
    return scored

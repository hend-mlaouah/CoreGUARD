from __future__ import annotations
import numpy as np
import pandas as pd


KPI_FAMILIES = {
    "pgw": "PGW — passerelle de données (sessions PDN, bearers EPS, plan usager)",
    "sgw": "SGW — passerelle de service (tunnels GTP, mobilité intra-LTE)",
    "pdc": "PDC — provisioning / exécution de commandes réseau",
    "stats": "Statistiques globales core network (PDP, EPS, RADIUS, DT)",
    "pmjob": "PM Job EPG — supervision matérielle (CPU, mémoire, GGSN)",
    "cal": "Contexte temporel (heure, jour de semaine, week-end)",
}


CAUSAL_RULES = {
    "pdc__avg_login_time": "Temps de login moyen anormalement élevé sur PDC — dégradation possible du provisioning ou de la charge du serveur de commandes.",
    "pdc__timeout_rate": "Taux de timeout de commandes élevé sur PDC — signe de saturation ou de latence excessive côté équipement provisionné.",
    "pdc__timeout-occured": "Occurrences de timeout en hausse sur PDC — à corréler avec la charge CPU/mémoire des cartes concernées.",
    "pdc__num-cmds-fail-timeout": "Nombre de commandes échouées par timeout en hausse — indique une instabilité du canal de provisioning.",
    "pgw__drop_rate_ul": "Taux de perte de paquets en liaison montante (UL) élevé sur PGW — congestion probable sur l'interface S1-U ou saturation du plan usager.",
    "pgw__uplink_dropped-packets__delta": "Volume de paquets uplink perdus en hausse sur PGW — à corréler avec le trafic total et la charge des bearers actifs.",
}


def kpi_family(feature_name: str) -> str:
    prefix = feature_name.split("__", 1)[0]
    return KPI_FAMILIES.get(prefix, "Autre / non catégorisé")


def causal_explanation(feature_name: str) -> str:
    base = feature_name.split("__roll_")[0].split("__lag_")[0]
    if base in CAUSAL_RULES:
        return CAUSAL_RULES[base]
    family = kpi_family(feature_name)
    return f"Valeur anormale sur cette métrique de la famille {family}."


def compute_shap_values(model, X: pd.DataFrame):
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):  
        shap_values = shap_values[-1]
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = expected_value[-1] if len(np.shape(expected_value)) else expected_value
    return shap_values, expected_value


def top_features_for_row(shap_values: np.ndarray, feature_names: list[str], row_idx: int, top_n: int = 5) -> pd.DataFrame:
    row_shap = shap_values[row_idx]
    order = np.argsort(-np.abs(row_shap))[:top_n]
    return pd.DataFrame({
        "feature": [feature_names[i] for i in order],
        "shap_value": [row_shap[i] for i in order],
        "family": [kpi_family(feature_names[i]) for i in order],
        "explication": [causal_explanation(feature_names[i]) for i in order],
    })


def global_feature_importance(shap_values: np.ndarray, feature_names: list[str], top_n: int = 20) -> pd.DataFrame:
    mean_abs = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({"feature": feature_names, "importance": mean_abs})
    df["family"] = df["feature"].apply(kpi_family)
    return df.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)


def build_rca_text(row_id, timestamp, severity: str, incident_proba: float, top_features_df: pd.DataFrame) -> str:
    lines = [
        f"Anomalie #{row_id} — {timestamp}",
        f"Sévérité : {severity} | Probabilité d'incident (XGBoost) : {incident_proba:.1%}",
        "",
        "Facteurs contributifs principaux (SHAP) :",
    ]
    for _, r in top_features_df.iterrows():
        direction = "▲ augmente" if r["shap_value"] > 0 else "▼ diminue"
        lines.append(f"  • {r['feature']} ({r['family']}) — {direction} le risque")
        lines.append(f"    → {r['explication']}")
    return "\n".join(lines)

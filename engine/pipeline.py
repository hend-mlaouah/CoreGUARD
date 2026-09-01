from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from engine.feature_engineering import build_engineered_features

ROLL_WINDOWS = {"1h": 4, "4h": 16, "24h": 96}
LAG_WINDOWS = {"15min": 1, "1h": 4, "24h": 96}
WARMUP_ROWS = 96 

CALENDAR_COLS = ["cal__heure", "cal__jour_semaine", "cal__est_weekend", "cal__heure_sin", "cal__heure_cos"]


@dataclass
class PipelineResult:
    engineered: pd.DataFrame              
    model_input: pd.DataFrame             
    warnings: list[str] = field(default_factory=list)
    used_local_scaler: bool = False
    n_rows_before_warmup_drop: int = 0
    n_rows_after_warmup_drop: int = 0


def add_rolling_lag_diff(F: pd.DataFrame) -> pd.DataFrame:
    out = F.copy()
    base_cols = [c for c in out.columns if c != "timestamp"]
    new_cols = {}
    for c in base_cols:
        s0 = pd.to_numeric(out[c], errors="coerce")
        for label, window in ROLL_WINDOWS.items():
            new_cols[f"{c}__roll_mean_{label}"] = s0.rolling(window).mean()
            new_cols[f"{c}__roll_std_{label}"] = s0.rolling(window).std()
        for label, shift in LAG_WINDOWS.items():
            new_cols[f"{c}__lag_{label}"] = s0.shift(shift)
        new_cols[f"{c}__diff_15min"] = s0.diff(1)
    return pd.concat([out, pd.DataFrame(new_cols, index=out.index)], axis=1)


def add_calendar_features(F: pd.DataFrame) -> pd.DataFrame:
    out = F.copy()
    ts = pd.to_datetime(out["timestamp"])
    out["cal__heure"] = ts.dt.hour
    out["cal__jour_semaine"] = ts.dt.dayofweek
    out["cal__est_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    out["cal__heure_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
    out["cal__heure_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24)
    return out


def apply_scaling(F: pd.DataFrame, scaler, warnings: list[str]) -> tuple[pd.DataFrame, bool]:
    out = F.copy()
    feature_cols = [c for c in out.columns if c != "timestamp" and c not in CALENDAR_COLS]

    if scaler is not None:
        try:
            scaler_features = list(getattr(scaler, "feature_names_in_", feature_cols))
            # Aligne strictement sur les colonnes attendues par le scaler,
            # dans son ordre exact.
            missing = [c for c in scaler_features if c not in out.columns]
            if missing:
                warnings.append(
                    f"{len(missing)} colonne(s) attendue(s) par le scaler absente(s) après "
                    f"feature engineering, remplies à 0 : {missing[:10]}{' ...' if len(missing) > 10 else ''}"
                )
            X = out.reindex(columns=scaler_features, fill_value=0.0).fillna(0.0)
            scaled = scaler.transform(X)
            out[scaler_features] = scaled
            return out, False
        except Exception as exc:
            warnings.append(f"Échec de l'application du scaler d'entraînement ({exc}) — normalisation locale utilisée à la place.")

    std = out[feature_cols].std(ddof=1).replace(0, 1.0)
    mean = out[feature_cols].mean()
    out[feature_cols] = (out[feature_cols] - mean) / std
    return out, True


def align_to_model_features(df: pd.DataFrame, feature_names: list[str], warnings: list[str]) -> pd.DataFrame:
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        warnings.append(
            f"{len(missing)} feature(s) attendue(s) par le modèle absente(s) des données "
            f"uploadées, remplies à 0 (impact potentiel sur la précision) : "
            f"{missing[:15]}{' ...' if len(missing) > 15 else ''}"
        )
    aligned = df.reindex(columns=feature_names, fill_value=0.0)
    aligned = aligned.replace([np.inf, -np.inf], np.nan)
    aligned = aligned.fillna(aligned.median(numeric_only=True)).fillna(0.0)
    return aligned


def run_pipeline(uploaded_files: dict, iso_forest_features: list[str], xgb_features: list[str], scaler=None) -> PipelineResult:
    warnings: list[str] = []

    raw_dfs = {}
    for source, file_obj in uploaded_files.items():
        if file_obj is None:
            raw_dfs[source] = None
            continue
        try:
            if hasattr(file_obj, "seek"):
                file_obj.seek(0)
            raw_dfs[source] = pd.read_csv(file_obj)
        except Exception as exc:
            warnings.append(f"[{source}] Erreur de lecture CSV : {exc}")
            raw_dfs[source] = None

    if all(v is None for v in raw_dfs.values()):
        empty = pd.DataFrame()
        return PipelineResult(engineered=empty, model_input=empty, warnings=warnings + ["Aucune source valide."])

    try:
        F = build_engineered_features(raw_dfs, warnings)
    except ValueError as exc:
        empty = pd.DataFrame()
        return PipelineResult(engineered=empty, model_input=empty, warnings=warnings + [str(exc)])

    F = add_rolling_lag_diff(F)
    F = add_calendar_features(F)
    F = F.replace([np.inf, -np.inf], np.nan)

    n_before = len(F)
    if len(F) > WARMUP_ROWS:
        F = F.iloc[WARMUP_ROWS:].reset_index(drop=True)
    else:
        warnings.append(
            f"Seulement {len(F)} ligne(s) après alignement sur la grille — moins que les "
            f"{WARMUP_ROWS} lignes de warm-up nécessaires pour les features 24h (comme à "
            f"l'entraînement). Les features __roll_*_24h / __lag_24h seront en grande partie "
            f"vides (remplies à 0/médiane), ce qui dégrade la fiabilité de la détection. "
            f"Uploade idéalement au moins 24h de données à 15 min de granularité."
        )
    n_after = len(F)

    F = F.set_index(pd.to_datetime(F["timestamp"])).drop(columns=["timestamp"])
    F.index.name = "timestamp"

    scaled, used_local_scaler = apply_scaling(F, scaler, warnings)

    reference_features = sorted(set(iso_forest_features) | set(xgb_features))
    model_input = align_to_model_features(scaled, reference_features, warnings)

    return PipelineResult(
        engineered=F,
        model_input=model_input,
        warnings=warnings,
        used_local_scaler=used_local_scaler,
        n_rows_before_warmup_drop=n_before,
        n_rows_after_warmup_drop=n_after,
    )

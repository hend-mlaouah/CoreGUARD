import re
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

INPUT_PATH = BASE_DIR / "data" / "processed" / "dataset_consolide.csv"   
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "dataset_features_reduites.csv"

COLS_NON_FEATURES = ["timestamp"]
SEUIL_VARIANCE_MIN = 0.01
SEUIL_CORRELATION = 0.9
FENETRE_A_GARDER = "1h"

LAG_A_GARDER = "15min"

KPIS_CRITIQUES = [
    "pdc__avg_login_time",
    "pdc__timeout_rate",
    "pdc__timeout-occured",
    "pdc__num-cmds-fail-timeout",
    "pgw__drop_rate_ul",
    "pgw__uplink_dropped-packets",
]


def charger_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    ts_col = df.columns[0] if "timestamp" not in df.columns else "timestamp"
    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.set_index(ts_col).sort_index()
    df.index.name = "timestamp"
    return df


def est_colonne_protegee(col: str) -> bool:
    """True si la colonne correspond a un KPI critique (colonne brute ou
    une de ses variantes __roll_mean_/__lag_/__diff_)."""
    return any(col == kpi or col.startswith(f"{kpi}__") for kpi in KPIS_CRITIQUES)


def filtre_1_reduire_fenetres_et_lags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retire les fenetres glissantes / lags qu'on ne veut pas garder,
    en se basant sur les suffixes generes par preprocessing_features.py
    (ex: '__roll_mean_4h', '__roll_mean_24h', '__lag_1h', '__lag_24h').
    Les colonnes protegees echappent aussi a ce filtre : on garde TOUTES
    leurs fenetres/lags, pas seulement 1h/15min.
    """
    a_dropper = []
    for col in df.columns:
        if est_colonne_protegee(col):
            continue
        if "__roll_mean_" in col or "__roll_std_" in col:
            if f"_{FENETRE_A_GARDER}" not in col:
                a_dropper.append(col)
        elif "__lag_" in col:
            if f"_{LAG_A_GARDER}" not in col:
                a_dropper.append(col)

    print(f"Filtre 1 - fenetres/lags non retenus supprimes : {len(a_dropper)}")
    return df.drop(columns=a_dropper)


def filtre_2_variance_minimale(df: pd.DataFrame) -> pd.DataFrame:
    """Retire les colonnes numeriques quasi constantes (peu d'info pour le modele),
    sauf les colonnes protegees (liste blanche KPIS_CRITIQUES)."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    variances = df[numeric_cols].var()
    a_dropper = [
        c for c in variances[variances < SEUIL_VARIANCE_MIN].index.tolist()
        if not est_colonne_protegee(c)
    ]
    proteges_epargnes = [
        c for c in variances[variances < SEUIL_VARIANCE_MIN].index.tolist()
        if est_colonne_protegee(c)
    ]

    print(f"Filtre 2 - colonnes a variance quasi nulle supprimees : {len(a_dropper)}")
    if a_dropper:
        print(a_dropper[:20], "..." if len(a_dropper) > 20 else "")
    if proteges_epargnes:
        print(f"Filtre 2 - colonnes protegees epargnees malgre variance faible : {proteges_epargnes}")
    return df.drop(columns=a_dropper)


def filtre_3_correlation_globale(df: pd.DataFrame, seuil: float) -> pd.DataFrame:

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr().abs()

    a_dropper = set()
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        if cols[i] in a_dropper:
            continue
        for j in range(i + 1, len(cols)):
            if cols[j] in a_dropper:
                continue
            if corr.iloc[i, j] > seuil:
                if est_colonne_protegee(cols[j]):
                    if not est_colonne_protegee(cols[i]):
                        a_dropper.add(cols[i])
                else:
                    a_dropper.add(cols[j])

    print(f"Filtre 3 - colonnes redondantes (correlation > {seuil}) supprimees : {len(a_dropper)}")
    return df.drop(columns=list(a_dropper))


def filtre_4_une_seule_variante_derivee(df: pd.DataFrame) -> pd.DataFrame:

    suffixe_roll = f"__roll_mean_{FENETRE_A_GARDER}"
    suffixe_lag = f"__lag_{LAG_A_GARDER}"

    colonnes = set(df.columns)
    a_dropper = []

    for col in df.columns:
        if est_colonne_protegee(col):
            continue
        if not col.endswith(suffixe_lag):
            continue
        base = col[: -len(suffixe_lag)]
        variante_roll = f"{base}{suffixe_roll}"
        if variante_roll in colonnes:
            a_dropper.append(col)

    print(f"Filtre 4 - variantes lag redondantes (roll_mean deja gardee) supprimees : {len(a_dropper)}")
    if a_dropper:
        print(a_dropper[:20], "..." if len(a_dropper) > 20 else "")
    return df.drop(columns=a_dropper)


def main():
    print(f"Chargement de {INPUT_PATH} ...")
    df = charger_dataset(INPUT_PATH)
    print(f"Shape initiale : {df.shape}\n")
    colonnes_critiques_absentes = [
        kpi for kpi in KPIS_CRITIQUES
        if not any(c == kpi or c.startswith(f"{kpi}__") for c in df.columns)
    ]
    if colonnes_critiques_absentes:
        print(f"[!] Attention : KPIs critiques absents des l'entree : {colonnes_critiques_absentes}\n")

    df = filtre_1_reduire_fenetres_et_lags(df)
    print(f"Shape apres filtre 1 : {df.shape}\n")

    df = filtre_2_variance_minimale(df)
    print(f"Shape apres filtre 2 : {df.shape}\n")

    df = filtre_3_correlation_globale(df, SEUIL_CORRELATION)
    print(f"Shape apres filtre 3 : {df.shape}\n")

    df = filtre_4_une_seule_variante_derivee(df)
    print(f"Shape apres filtre 4 : {df.shape}\n")

    print("=" * 70)
    print(f"Shape finale : {df.shape}")
    print(f"Colonnes conservees : {list(df.columns)}")
    print("=" * 70)

    # Sanity-check final
    colonnes_critiques_conservees = [
        kpi for kpi in KPIS_CRITIQUES
        if any(c == kpi or c.startswith(f"{kpi}__") for c in df.columns)
    ]
    print(f"\nKPIs critiques conserves dans la sortie : {colonnes_critiques_conservees}")

    OUTPUT_PATH.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(OUTPUT_PATH)
    print(f"\nDataset reduit sauvegarde : {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
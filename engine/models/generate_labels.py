from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent   
PROCESSED_DIR = BASE_DIR / "data" / "processed"

FEATURES_PATH = PROCESSED_DIR / "dataset_features_reduites.csv"
OUTPUT_PATH = PROCESSED_DIR / "dataset_labels.csv"

KPIS_A_VERIFIER = [
    "pdc__avg_login_time",
    "pdc__num-cmds-fail-timeout",
    "pdc__timeout_rate",
    "pdc__timeout-occured",
    "pgw__drop_rate_ul",
    "pgw__uplink_dropped-packets__delta",
]

SEUIL_PERCENTILE = 0.95
NB_KPIS_MIN_SEUIL = 2
TOLERANCE_CRENEAUX = 1


def main():
    print(f"Chargement de {FEATURES_PATH} ...")
    df = pd.read_csv(FEATURES_PATH, index_col=0)
    df.index = pd.to_datetime(df.index)
    print(f"Shape : {df.shape}")

    outliers_par_kpi = {}
    for kpi in KPIS_A_VERIFIER:
        if kpi not in df.columns:
            print(f"[!] Colonne introuvable, ignoree : {kpi}")
            continue
        seuil = df[kpi].quantile(SEUIL_PERCENTILE)
        outliers_par_kpi[kpi] = set(df.index[df[kpi] > seuil])

    compteur = {}
    for kpi, timestamps in outliers_par_kpi.items():
        for t in timestamps:
            compteur[t] = compteur.get(t, 0) + 1

    timestamps_incident = {t for t, n in compteur.items() if n >= NB_KPIS_MIN_SEUIL}
    print(f"\nTimestamps marques incident=1 (>= {NB_KPIS_MIN_SEUIL} KPIs simultanes) : "
          f"{len(timestamps_incident)} / {len(df)} ({100*len(timestamps_incident)/len(df):.1f}%)")

    df["label_incident"] = df.index.isin(timestamps_incident).astype(int)

    print("\nRepartition des labels :")
    print(df["label_incident"].value_counts())

    df.to_csv(OUTPUT_PATH)
    print(f"\nDataset avec labels sauvegarde : {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
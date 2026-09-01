import logging
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUT_DIR = BASE_DIR / "data" / "cleaned"          
REPORT_DIR = BASE_DIR / "outputs" / "reports"     



def drop_constant_columns(df, exclude=()):
    value_cols = [c for c in df.columns if c not in exclude]
    constant_cols = [c for c in value_cols if df[c].nunique(dropna=False) <= 1]
    return df.drop(columns=constant_cols), constant_cols


def convert_to_delta(df, cumulative_cols):
    reset_summary = {}
    for c in cumulative_cols:
        delta = df[c].diff()
        n_reset = (delta < 0).sum()
        if n_reset > 0:
            reset_summary[c] = n_reset
        df[c + "__delta"] = delta.clip(lower=0)
    return df.drop(columns=cumulative_cols), reset_summary


def save_with_report(df, output_path, report, report_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    with open(report_path, "w") as f:
        f.write("\n".join(report))



def clean_pgw():
    df = pd.read_csv(RAW_DIR / "pdc_pgw_statistics.csv", sep="|")
    report = [f"[PGW] Input shape: {df.shape}"]

    ts_cols = [c for c in ("Time", "time-started", "time-sampled") if c in df.columns]
    for c in ts_cols:
        df[c] = pd.to_datetime(df[c])
    df = df.sort_values("Time")

    df, constant_cols = drop_constant_columns(df, exclude=ts_cols)
    report.append(f"Constant columns dropped: {len(constant_cols)}")

    value_cols = [c for c in df.columns if c not in ts_cols]
    cumulative, gauges = [], []
    for c in value_cols:
        diffs = df[c].diff().dropna()
        frac_neg = (diffs < 0).mean() if len(diffs) else 0
        (cumulative if frac_neg < 0.02 else gauges).append(c)

    df, reset_summary = convert_to_delta(df, cumulative)
    report.append(f"Gauge: {len(gauges)}, Cumulative->delta: {len(cumulative)}, Resets: {len(reset_summary)}")
    df = df.dropna().reset_index(drop=True)
    df = df.drop(columns=["time-started"])

    save_with_report(df, OUT_DIR / "pgw_cleaned.csv", report, REPORT_DIR / "pgw_report.txt")


def clean_sgw():
    df = pd.read_csv(RAW_DIR / "pdc_sgw_statistics.csv", sep="|")
    report = [f"[SGW] Input shape: {df.shape}"]

    ts_cols = [c for c in ("Time", "time-started", "time-sampled") if c in df.columns]
    for c in ts_cols:
        df[c] = pd.to_datetime(df[c])
    df = df.sort_values("Time")

    df, constant_cols = drop_constant_columns(df, exclude=ts_cols)
    report.append(f"Constant columns dropped: {len(constant_cols)}")

    never_delta = [c for c in df.columns if any(
        kw in c.lower() for kw in ("buffer", "gtp-state", "s1u", "s5s8")
    )]

    value_cols = [c for c in df.columns if c not in ts_cols and c not in never_delta]
    cumulative, gauges = [], []
    for c in value_cols:
        diffs = df[c].diff().dropna()
        frac_neg = (diffs < 0).mean() if len(diffs) else 0
        (cumulative if frac_neg < 0.02 else gauges).append(c)

    df, reset_summary = convert_to_delta(df, cumulative)
    report.append(f"Gauge: {len(gauges)}, Cumulative->delta: {len(cumulative)}, "
                  f"Exclues (never_delta): {len(never_delta)}, Resets: {len(reset_summary)}")
    df = df.dropna().reset_index(drop=True)

    save_with_report(df, OUT_DIR / "sgw_cleaned.csv", report, REPORT_DIR / "sgw_report.txt")


def clean_statistics():
    df = pd.read_csv(RAW_DIR / "pdc_statistics.csv", sep="|")
    report = [f"[STATISTICS] Input shape: {df.shape}"]

    ts_cols = [c for c in ("Time",) if c in df.columns]
    for c in ts_cols:
        df[c] = pd.to_datetime(df[c])
    df = df.sort_values("Time") if "Time" in df.columns else df

    df, constant_cols = drop_constant_columns(df, exclude=ts_cols)
    report.append(f"Constant columns dropped: {len(constant_cols)}")
    df = df.dropna().reset_index(drop=True)

    save_with_report(df, OUT_DIR / "statistics_cleaned.csv", report, REPORT_DIR / "statistics_report.txt")


def clean_pdc():
    df = pd.read_csv(RAW_DIR / "pdc-data.csv", sep="|")
    report = [f"[PDC] Input shape: {df.shape}"]

    ts_cols = ["Time"]
    df["Time"] = pd.to_datetime(df["Time"])
    for c in ts_cols:
        df[c] = pd.to_datetime(df[c], errors="coerce")

    
    df, constant_cols = drop_constant_columns(df, exclude=ts_cols)
    report.append(f"Constant columns dropped: {len(constant_cols)}")
    df = df.dropna(how="all").reset_index(drop=True)

    save_with_report(df, OUT_DIR / "pdc_cleaned.csv", report, REPORT_DIR / "pdc_report.txt")


def clean_pm():

    df = pd.read_csv(RAW_DIR / "pm_job_epg-all.csv", sep="|")

    report = []
    report.append("=" * 60)
    report.append("PM JOB CLEANING REPORT")
    report.append("=" * 60)

    report.append(f"Input shape: {df.shape}")

    ts_cols = []

    for c in df.columns:
        if "time" in c.lower():
            try:
                df[c] = pd.to_datetime(df[c], errors="coerce")
                ts_cols.append(c)
            except:
                pass


    if "Time" in df.columns:
        n_before = len(df)
        bad_time_mask = df["Time"].isna()
        n_bad = bad_time_mask.sum()
        if n_bad > 0:
            report.append(
                f"Malformed rows dropped (Time unparsable, likely duplicated header): {n_bad}"
            )
            df = df.loc[~bad_time_mask].reset_index(drop=True)
        report.append(f"Rows after malformed-row removal: {len(df)} (was {n_before})")

        df = df.sort_values("Time").reset_index(drop=True)


    header_echo_mask = pd.Series(False, index=df.index)
    for c in df.columns:
        if c in ts_cols:
            continue
        header_echo_mask |= (df[c].astype(str) == c)
    n_echo = header_echo_mask.sum()
    if n_echo > 0:
        report.append(f"Additional header-echo rows dropped: {n_echo}")
        df = df.loc[~header_echo_mask].reset_index(drop=True)

    dup_rows = df.duplicated().sum()
    report.append(f"Duplicate rows: {dup_rows}")

    if "Time" in df.columns:
        dup_time = df["Time"].duplicated().sum()
        report.append(f"Duplicate Time values: {dup_time}")

    empty_cols = df.columns[df.isna().all()].tolist()
    df = df.drop(columns=empty_cols)
    report.append(f"Empty columns dropped: {len(empty_cols)}")

    df, constant_cols = drop_constant_columns(df, exclude=ts_cols)
    report.append(f"Constant columns dropped: {len(constant_cols)}")


    numeric_cols = []
    for c in df.columns:
        if c not in ts_cols:
            converted = pd.to_numeric(df[c], errors="coerce")
            n_new_na = converted.isna().sum() - df[c].isna().sum()
            if n_new_na > 0:
                report.append(
                    f"  WARNING: {c} — {n_new_na} valeur(s) non numériques converties en NaN"
                )
            df[c] = converted
            if pd.api.types.is_numeric_dtype(df[c]):
                numeric_cols.append(c)

    report.append(f"Numeric KPI columns: {len(numeric_cols)}")

    df = df.replace([float("inf"), float("-inf")], pd.NA)
    missing = df.isna().sum().sum()
    report.append(f"Total missing values: {missing}")
    if "Time" in df.columns:
        interval = df["Time"].diff().dropna()
        if len(interval):
            report.append(f"Sampling interval median: {interval.median()}")
            report.append(f"Sampling interval min: {interval.min()}")
            report.append(f"Sampling interval max: {interval.max()}")


    report.append(f"Final shape: {df.shape}")
    save_with_report(
        df,
        OUT_DIR / "pm_job_epg-all_cleaned.csv",
        report,
        REPORT_DIR / "pm_job_report.txt"
    )


def main():
    for fn in (clean_pgw, clean_sgw, clean_statistics, clean_pdc, clean_pm):
        try:
            fn()
            logger.info(f"[OK] {fn.__name__}")
        except FileNotFoundError as e:
            logger.warning(f"[SKIP] {fn.__name__}: {e}")
        except Exception as e:
            logger.error(f"[FAIL] {fn.__name__}: {e}")

    logger.info(f"[CHECK] Fichiers cleaned générés dans {OUT_DIR.resolve()} :")
    for f in sorted(OUT_DIR.glob("*.csv")):
        logger.info(f"  - {f.name}")


if __name__ == "__main__":
    main()
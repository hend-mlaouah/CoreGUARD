from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib

BASE = Path(__file__).resolve().parent.parent.parent  
CLEANED_DIR = BASE / "data" / "cleaned"
PROCESSED_DIR = BASE / "data" / "processed"
MODELS_DIR = BASE / "models"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = PROCESSED_DIR / "dataset_consolide.csv"
SCALER_PATH = MODELS_DIR / "scaler.pkl"


print(f"[LOAD] pgw   <- {(CLEANED_DIR / 'pgw_cleaned.csv').resolve()}")
pgw = pd.read_csv(CLEANED_DIR / "pgw_cleaned.csv", low_memory=False)

print(f"[LOAD] sgw   <- {(CLEANED_DIR / 'sgw_cleaned.csv').resolve()}")
sgw = pd.read_csv(CLEANED_DIR / "sgw_cleaned.csv", low_memory=False)

print(f"[LOAD] pdc   <- {(CLEANED_DIR / 'pdc_cleaned.csv').resolve()}")
pdc = pd.read_csv(CLEANED_DIR / "pdc_cleaned.csv", low_memory=False)

print(f"[LOAD] stats <- {(CLEANED_DIR / 'statistics_cleaned.csv').resolve()}")
stats = pd.read_csv(CLEANED_DIR / "statistics_cleaned.csv", low_memory=False)

print(f"[LOAD] pmjob <- {(CLEANED_DIR / 'pm_job_epg-all_cleaned.csv').resolve()}")
pm = pd.read_csv(CLEANED_DIR / "pm_job_epg-all_cleaned.csv", low_memory=False)

grid = pd.date_range(
    start="2026-06-02 11:45:00",
    end="2026-07-02 09:45:00",
    freq="15min"
)


def nearest_to_grid(df, time_col="Time"):
    x = df.copy()
    x["_ts"] = pd.to_datetime(x[time_col])
    x = x.sort_values("_ts")
    q = pd.DataFrame({"_ts": grid})
    out = pd.merge_asof(
        q, x, on="_ts",
        direction="nearest",
        tolerance=pd.Timedelta("7min")
    )
    return out


A = {
    "pgw": nearest_to_grid(pgw),
    "sgw": nearest_to_grid(sgw),
    "pdc": nearest_to_grid(pdc),
    "stats": nearest_to_grid(stats),
    "pmjob": nearest_to_grid(pm),
}


def num(s):
    return pd.to_numeric(s, errors="coerce")


def safe_div(a, b):
    return num(a) / num(b).replace(0, np.nan)


# ---------------------------------------------------------------------------
# FEATURE ENGINEERING PAR SOURCE
# ---------------------------------------------------------------------------
g = A["pgw"]
s = A["sgw"]
d = A["pdc"]
st = A["stats"]
pm0 = A["pmjob"]

F = pd.DataFrame({"timestamp": grid})

# PGW
pgw_direct = [
    "subscriber-count", "pdp-active", "eps-active-bearer",
    "eps-active-ipv6-bearer", "pdp-created__delta",
    "pdp-create-attempted__delta", "pdp-create-failed__delta",
    "eps-bearer-creation__delta", "eps-bearer-creation-attempted__delta",
    "eps-bearer-creation-failed__delta", "uplink_bytes__delta",
    "downlink_bytes__delta", "uplink_packets__delta",
    "downlink_packets__delta", "uplink_dropped-packets__delta",
    "downlink_dropped-packets__delta", "pdn-suspended-connections",
    "pdn-pgw-connections",
]
for c in pgw_direct:
    F[f"pgw__{c}"] = num(g[c])

F["pgw__pdp_success_rate"] = safe_div(g["pdp-created__delta"], g["pdp-create-attempted__delta"])
F["pgw__pdp_failure_rate"] = safe_div(g["pdp-create-failed__delta"], g["pdp-create-attempted__delta"])
F["pgw__eps_success_rate"] = safe_div(g["eps-bearer-creation__delta"], g["eps-bearer-creation-attempted__delta"])
F["pgw__eps_failure_rate"] = safe_div(g["eps-bearer-creation-failed__delta"], g["eps-bearer-creation-attempted__delta"])
F["pgw__total_traffic"] = num(g["uplink_bytes__delta"]) + num(g["downlink_bytes__delta"])
F["pgw__ul_dl_traffic_ratio"] = safe_div(g["uplink_bytes__delta"], g["downlink_bytes__delta"])
F["pgw__total_packets"] = num(g["uplink_packets__delta"]) + num(g["downlink_packets__delta"])
F["pgw__drop_rate_ul"] = safe_div(g["uplink_dropped-packets__delta"], g["uplink_packets__delta"])
F["pgw__drop_rate_dl"] = safe_div(g["downlink_dropped-packets__delta"], g["downlink_packets__delta"])
F["pgw__suspended_connection_ratio"] = safe_div(g["pdn-suspended-connections"], g["pdn-pgw-connections"])

# SGW
sgw_direct = [
    "sgwGtpTunnelMgmtS4S11-smCreateSessionReqRcvd__delta",
    "sgwGtpTunnelMgmtS4S11-smCreateSessionRespAccSent__delta",
    "sgwGtpTunnelMgmtS4S11-smCreateSessionRespRejSent__delta",
    "sgwGtpTunnelMgmtS4S11-smModifyBearerReqRcvd__delta",
    "sgwGtpTunnelMgmtS4S11-smModifyBearerRespRejSent__delta",
    "sgwGtpTrafficS5S8-inDataByte",
    "sgwGtpTrafficS5S8-outDataByte",
    "sgwGtpTrafficS1uS4S12-inDataByte",
    "sgwGtpTrafficS1uS4S12-outDataByte",
    "sgwUplinkTraffic-sgwUplinkDroppedPackets__delta",
    "sgwDownlinkTraffic-sgwDownlinkDroppedPackets__delta",
    "sgwGtpTrafficS5S8-inDataPkt",
    "sgwGtpTrafficS5S8-outDataPkt",
    "sgwGtpTrafficS1uS4S12-inDataPkt",
    "sgwGtpTrafficS1uS4S12-outDataPkt",
]
for c in sgw_direct:
    F[f"sgw__{c}"] = num(s[c])

F["sgw__session_success_rate"] = safe_div(
    s["sgwGtpTunnelMgmtS4S11-smCreateSessionRespAccSent__delta"],
    s["sgwGtpTunnelMgmtS4S11-smCreateSessionReqRcvd__delta"]
)
F["sgw__session_failure_rate"] = safe_div(
    s["sgwGtpTunnelMgmtS4S11-smCreateSessionRespRejSent__delta"],
    s["sgwGtpTunnelMgmtS4S11-smCreateSessionReqRcvd__delta"]
)
F["sgw__bearer_modification_failure_rate"] = safe_div(
    s["sgwGtpTunnelMgmtS4S11-smModifyBearerRespRejSent__delta"],
    s["sgwGtpTunnelMgmtS4S11-smModifyBearerReqRcvd__delta"]
)

F["sgw__total_traffic"] = (
    num(s["sgwGtpTrafficS5S8-outDataByte"])
    + num(s["sgwGtpTrafficS1uS4S12-outDataByte"])
)
F["sgw__packet_drop_rate"] = safe_div(
    num(s["sgwUplinkTraffic-sgwUplinkDroppedPackets__delta"])
    + num(s["sgwDownlinkTraffic-sgwDownlinkDroppedPackets__delta"]),
    num(s["sgwGtpTrafficS5S8-inDataPkt"])
    + num(s["sgwGtpTrafficS5S8-outDataPkt"])
    + num(s["sgwGtpTrafficS1uS4S12-inDataPkt"])
    + num(s["sgwGtpTrafficS1uS4S12-outDataPkt"])
)
F["sgw__suspended_ue_ratio"] = safe_div(
    s["sgwNumberOfUes-nbrOfSuspendedUes"],
    s["sgwNumberOfUes-sgwNbrOfUes"]
)
F["sgw__idle_bearer_ratio"] = safe_div(
    s["sgwNumberOfSessions-sgwNbrOfIdleBearers"],
    s["sgwNumberOfSessions-sgwNbrOfBearers"]
)
F["sgw__connected_ue_ratio"] = safe_div(
    s["sgwNumberOfUes-sgwNbrOfConnectedUes"],
    s["sgwNumberOfUes-sgwNbrOfUes"]
)

# PDC
pdc_direct = [
    "tot-run-time", "num-cmds-ok", "num-cmds-fail-timeout",
    "num-cmds-default-ok", "num-cmds-mapn-ok", "num-cmds-aapn-ok",
    "time-default", "time-mapn", "time-aapn", "login-time",
    "timeout-occured", "session-backout-fail"
]
for c in pdc_direct:
    F[f"pdc__{c}"] = num(d[c])

cmd_total = num(d["num-cmds-ok"]) + num(d["num-cmds-fail-timeout"])
F["pdc__timeout_rate"] = safe_div(d["num-cmds-fail-timeout"], cmd_total)
F["pdc__command_success_rate"] = safe_div(d["num-cmds-ok"], cmd_total)
F["pdc__avg_runtime_per_command"] = safe_div(d["tot-run-time"], cmd_total)
# Valeur brute (pas de division) — cohérent avec le pipeline d'inférence.
F["pdc__avg_login_time"] = num(d["login-time"])
F["pdc__configuration_time"] = (
    num(d["time-default"]) + num(d["time-mapn"]) + num(d["time-aapn"])
)
F["pdc__mapn_ratio"] = safe_div(d["num-cmds-mapn-ok"], cmd_total)
F["pdc__aapn_ratio"] = safe_div(d["num-cmds-aapn-ok"], cmd_total)
F["pdc__default_command_ratio"] = safe_div(d["num-cmds-default-ok"], cmd_total)

# Statistics
stats_direct = [
    "Active PDP contexts", "Active EPS bearers", "PDP creations", " Failed",
    "EPS bearer creations", " Failed.3", "PDP updates", " Failed.7",
    "EPS bearer modifications", " Failed.8", "PDP deactivations", " Failed.9",
    "Packets", "Bytes", "Dropped packets", "Packets.1", "Bytes.1",
    "Dropped packets.1", "Active subscribers",
    "Failed RADIUS Accounting procedures",
    "Total successful DT establishments", "Total requests for DT establishments",
    "DT RNC error indications"
]
for c in stats_direct:
    F[f"stats__{c}"] = num(st[c])

F["stats__pdp_success_rate"] = safe_div(st["PDP creations"], num(st["PDP creations"]) + num(st[" Failed"]))
F["stats__pdp_failure_rate"] = safe_div(st[" Failed"], num(st["PDP creations"]) + num(st[" Failed"]))
F["stats__eps_success_rate"] = safe_div(st["EPS bearer creations"], num(st["EPS bearer creations"]) + num(st[" Failed.3"]))
F["stats__eps_failure_rate"] = safe_div(st[" Failed.3"], num(st["EPS bearer creations"]) + num(st[" Failed.3"]))
F["stats__update_failure_rate"] = safe_div(st[" Failed.7"], num(st["PDP updates"]) + num(st[" Failed.7"]))
F["stats__modification_failure_rate"] = safe_div(st[" Failed.8"], num(st["EPS bearer modifications"]) + num(st[" Failed.8"]))
F["stats__deactivation_failure_rate"] = safe_div(st[" Failed.9"], num(st["PDP deactivations"]) + num(st[" Failed.9"]))
F["stats__total_traffic"] = num(st["Bytes"]) + num(st["Bytes.1"])
F["stats__total_packets"] = num(st["Packets"]) + num(st["Packets.1"])
F["stats__total_dropped_packets"] = num(st["Dropped packets"]) + num(st["Dropped packets.1"])
F["stats__packet_drop_rate"] = safe_div(
    num(st["Dropped packets"]) + num(st["Dropped packets.1"]),
    num(st["Packets"]) + num(st["Packets.1"])
)
F["stats__avg_packet_size"] = safe_div(
    num(st["Bytes"]) + num(st["Bytes.1"]),
    num(st["Packets"]) + num(st["Packets.1"])
)
F["stats__traffic_per_subscriber"] = safe_div(
    num(st["Bytes"]) + num(st["Bytes.1"]),
    st["Active subscribers"]
)
F["stats__pdp_per_subscriber"] = safe_div(
    st["Active PDP contexts"], st["Active subscribers"]
)
F["stats__eps_per_subscriber"] = safe_div(
    st["Active EPS bearers"], st["Active subscribers"]
)
F["stats__dt_success_rate"] = safe_div(
    st["Total successful DT establishments"],
    st["Total requests for DT establishments"]
)
F["stats__dt_error_rate"] = safe_div(
    st["DT RNC error indications"],
    st["Total requests for DT establishments"]
)
F["stats__radius_failure_rate"] = safe_div(
    st["Failed RADIUS Accounting procedures"], st["Active subscribers"]
)

# PM/EPG
pm_num = pm0.drop(columns=["Time", "_ts"], errors="ignore").apply(pd.to_numeric, errors="coerce")
cpu_avg = [c for c in pm_num.columns if "average-cpu-usage" in c]
cpu_peak = [c for c in pm_num.columns if "peak-cpu-usage" in c]
mem = [c for c in pm_num.columns if ":memory:" in c]
mem_used = [c for c in pm_num.columns if ":memory-used:" in c]
gtp_errors = [c for c in pm_num.columns if c.startswith("ggsn-gtp-error-stats:")]

F["pmjob__cpu_mean"] = pm_num[cpu_avg].mean(axis=1)
F["pmjob__cpu_max"] = pm_num[cpu_peak].max(axis=1)
F["pmjob__memory_usage_rate"] = safe_div(pm_num[mem_used].sum(axis=1), pm_num[mem].sum(axis=1))
F["pmjob__pdp_success_rate"] = safe_div(
    pm_num["ggsn-pdp-contexts-stats-completed:ggsn-completed-activation"],
    pm_num["ggsn-pdp-contexts-stats-attempted:ggsn-attempted-activation"]
)
F["pmjob__pdp_failure_rate"] = safe_div(
    pm_num["ggsn-pdp-contexts-stats-failed:ggsn-failed-activation"],
    pm_num["ggsn-pdp-contexts-stats-attempted:ggsn-attempted-activation"]
)
F["pmjob__gtp_error_rate"] = safe_div(
    pm_num[gtp_errors].sum(axis=1),
    pm_num["ggsn-gtp-stats:ggsn-gtp-requests-accepted"]
)
F["pmjob__total_bytes"] = (
    pm_num["ggsn-uplink-traffic-info:ggsn-uplink-bytes"]
    + pm_num["ggsn-downlink-traffic-info:ggsn-downlink-bytes"]
)
F["pmjob__total_packets"] = (
    pm_num["ggsn-uplink-traffic-info:ggsn-uplink-packets"]
    + pm_num["ggsn-downlink-traffic-info:ggsn-downlink-packets"]
)
F["pmjob__drop_rate"] = safe_div(
    pm_num["ggsn-uplink-traffic-info:ggsn-uplink-drops"]
    + pm_num["ggsn-downlink-traffic-info:ggsn-downlink-drops"],
    pm_num["ggsn-uplink-traffic-info:ggsn-uplink-packets"]
    + pm_num["ggsn-downlink-traffic-info:ggsn-downlink-packets"]
)
F["pmjob__tunnel_utilization"] = safe_div(
    pm_num["ggsn-gtp-stats:ggsn-gtp-nbr-of-created-tunnels"],
    pm_num["ggsn-gtp-stats:ggsn-gtp-nbr-of-tunnels"]
)
F["pmjob__pdp_per_subscriber"] = safe_div(
    pm_num["ggsn-global-stats:ggsn-nbr-of-active-pdp-contexts"],
    pm_num["ggsn-global-stats:ggsn-nbr-of-subscribers"]
)

base_cols = [c for c in F.columns if c != "timestamp"]

for c in base_cols:
    s0 = pd.to_numeric(F[c], errors="coerce")
    F[f"{c}__roll_mean_1h"] = s0.rolling(4).mean()
    F[f"{c}__roll_std_1h"] = s0.rolling(4).std()
    F[f"{c}__roll_mean_4h"] = s0.rolling(16).mean()
    F[f"{c}__roll_std_4h"] = s0.rolling(16).std()
    F[f"{c}__roll_mean_24h"] = s0.rolling(96).mean()
    F[f"{c}__roll_std_24h"] = s0.rolling(96).std()
    F[f"{c}__lag_15min"] = s0.shift(1)
    F[f"{c}__lag_1h"] = s0.shift(4)
    F[f"{c}__lag_24h"] = s0.shift(96)
    F[f"{c}__diff_15min"] = s0.diff(1)


feature_cols = [c for c in F.columns if c != "timestamp"]
for c in feature_cols:
    F[c] = pd.to_numeric(F[c], errors="coerce")

means = F[feature_cols].mean()
stds = F[feature_cols].std(ddof=1)

F_scaled = F.copy()
F_scaled[feature_cols] = (F[feature_cols] - means) / stds

ts = pd.to_datetime(F_scaled["timestamp"])
F_scaled["cal__heure"] = ts.dt.hour
F_scaled["cal__jour_semaine"] = ts.dt.dayofweek
F_scaled["cal__est_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
F_scaled["cal__heure_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
F_scaled["cal__heure_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24)

rebuilt = F_scaled.iloc[96:].reset_index(drop=True)
rebuilt.to_csv(OUTPUT_PATH, index=False)
print(f"[SAVE] dataset_consolide -> {OUTPUT_PATH.resolve()}")
print(f"  Shape : {rebuilt.shape}")

scaler = StandardScaler()
scaler.mean_ = means.to_numpy(dtype=float)
scaler.scale_ = stds.to_numpy(dtype=float)
scaler.var_ = stds.to_numpy(dtype=float) ** 2
scaler.n_features_in_ = len(feature_cols)
scaler.feature_names_in_ = np.array(feature_cols, dtype=object)
joblib.dump(scaler, SCALER_PATH)
print(f"[SAVE] scaler -> {SCALER_PATH.resolve()}")

print("\nTerminé.")
import argparse
import os
import re
import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None

def safe_div(numerator, denominator):
    """Division sécurisée : évite les division par zéro / NaN -> NaN."""
    num = numerator.astype(float)
    den = denominator.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = num / den
    result = result.replace([np.inf, -np.inf], np.nan)
    return result


def keep_existing(df, cols):
    """Retourne uniquement les colonnes de `cols` qui existent réellement dans df."""
    existing = [c for c in cols if c in df.columns]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"  [!] Colonnes absentes (ignorées) : {missing}")
    return existing


# ---------------------------------------------------------------------------
# 1. PGW
# ---------------------------------------------------------------------------
def process_pgw(path):
    print("\n=== PGW ===")
    df = pd.read_csv(path)

    base_kpis = [
        "subscriber-count",
        "pdp-active",
        "eps-active-bearer",
        "eps-active-ipv6-bearer",
    ]

    formula_cols = [
        "pdp-created__delta",
        "pdp-create-attempted__delta",
        "pdp-create-failed__delta",
        "eps-bearer-creation__delta",
        "eps-bearer-creation-attempted__delta",
        "eps-bearer-creation-failed__delta",
        "uplink_bytes__delta",
        "downlink_bytes__delta",
        "uplink_packets__delta",
        "downlink_packets__delta",
        "uplink_dropped-packets__delta",
        "downlink_dropped-packets__delta",
        "pdn-suspended-connections",
        "pdn-pgw-connections",  
    ]

    keep_cols =  ["Time"] + keep_existing(df, base_kpis + formula_cols)
    out = df[keep_cols].copy()

    # --- Nouvelles features ---
    out["pdp_success_rate"] = safe_div(df["pdp-created__delta"], df["pdp-create-attempted__delta"])
    out["pdp_failure_rate"] = safe_div(df["pdp-create-failed__delta"], df["pdp-create-attempted__delta"])
    out["eps_success_rate"] = safe_div(df["eps-bearer-creation__delta"], df["eps-bearer-creation-attempted__delta"])
    out["eps_failure_rate"] = safe_div(df["eps-bearer-creation-failed__delta"], df["eps-bearer-creation-attempted__delta"])

    out["total_traffic"] = df["uplink_bytes__delta"] + df["downlink_bytes__delta"]
    out["ul_dl_traffic_ratio"] = safe_div(df["uplink_bytes__delta"], df["downlink_bytes__delta"])
    out["total_packets"] = df["uplink_packets__delta"] + df["downlink_packets__delta"]
    out["drop_rate_ul"] = safe_div(df["uplink_dropped-packets__delta"], df["uplink_packets__delta"])
    out["drop_rate_dl"] = safe_div(df["downlink_dropped-packets__delta"], df["downlink_packets__delta"])
    out["suspended_connection_ratio"] = safe_div(df["pdn-suspended-connections"], df["pdn-pgw-connections"])

    print(f"  -> {out.shape[1]} colonnes conservées / créées, {out.shape[0]} lignes")
    return out


# ---------------------------------------------------------------------------
# 2. SGW
# ---------------------------------------------------------------------------
def process_sgw(path):
    print("\n=== SGW ===")
    df = pd.read_csv(path)


    formula_cols = [
        # Session (interface S4/S11 : requêtes reçues par le SGW)
        "sgwGtpTunnelMgmtS4S11-smCreateSessionReqRcvd__delta",
        "sgwGtpTunnelMgmtS4S11-smCreateSessionRespAccSent__delta",
        "sgwGtpTunnelMgmtS4S11-smCreateSessionRespRejSent__delta",
        # Bearer modification
        "sgwGtpTunnelMgmtS4S11-smModifyBearerReqRcvd__delta",
        "sgwGtpTunnelMgmtS4S11-smModifyBearerRespRejSent__delta",
        # Traffic
        "sgwGtpTrafficS5S8-inDataByte",
        "sgwGtpTrafficS5S8-outDataByte",
        "sgwGtpTrafficS1uS4S12-inDataByte",
        "sgwGtpTrafficS1uS4S12-outDataByte",
        # Packet drops
        "sgwUplinkTraffic-sgwUplinkDroppedPackets__delta",
        "sgwDownlinkTraffic-sgwDownlinkDroppedPackets__delta",
        "sgwGtpTrafficS5S8-inDataPkt",
        "sgwGtpTrafficS5S8-outDataPkt",
        "sgwGtpTrafficS1uS4S12-inDataPkt",
        "sgwGtpTrafficS1uS4S12-outDataPkt",
    ]

    keep_cols = ["Time"] + keep_existing(df,  formula_cols)
    out = df[keep_cols].copy()

    out["session_success_rate"] = safe_div(
        df["sgwGtpTunnelMgmtS4S11-smCreateSessionRespAccSent__delta"],
        df["sgwGtpTunnelMgmtS4S11-smCreateSessionReqRcvd__delta"],
    )
    out["session_failure_rate"] = safe_div(
        df["sgwGtpTunnelMgmtS4S11-smCreateSessionRespRejSent__delta"],
        df["sgwGtpTunnelMgmtS4S11-smCreateSessionReqRcvd__delta"],
    )
    out["bearer_modification_failure_rate"] = safe_div(
        df["sgwGtpTunnelMgmtS4S11-smModifyBearerRespRejSent__delta"],
        df["sgwGtpTunnelMgmtS4S11-smModifyBearerReqRcvd__delta"],
    )

    out["total_traffic"] = (
        df["sgwGtpTrafficS5S8-inDataByte"]
        + df["sgwGtpTrafficS5S8-outDataByte"]
        + df["sgwGtpTrafficS1uS4S12-inDataByte"]
        + df["sgwGtpTrafficS1uS4S12-outDataByte"]
    )

    total_packets = (
        df["sgwGtpTrafficS5S8-inDataPkt"]
        + df["sgwGtpTrafficS5S8-outDataPkt"]
        + df["sgwGtpTrafficS1uS4S12-inDataPkt"]
        + df["sgwGtpTrafficS1uS4S12-outDataPkt"]
    )
    dropped_packets = (
        df["sgwUplinkTraffic-sgwUplinkDroppedPackets__delta"]
        + df["sgwDownlinkTraffic-sgwDownlinkDroppedPackets__delta"]
    )
    out["packet_drop_rate"] = safe_div(dropped_packets, total_packets)

    out["suspended_ue_ratio"] = safe_div(
        df["sgwNumberOfUes-nbrOfSuspendedUes"], df["sgwNumberOfUes-sgwNbrOfUes"]
    )
    out["idle_bearer_ratio"] = safe_div(
        df["sgwNumberOfSessions-sgwNbrOfIdleBearers"], df["sgwNumberOfSessions-sgwNbrOfBearers"]
    )
    out["connected_ue_ratio"] = safe_div(
        df["sgwNumberOfUes-sgwNbrOfConnectedUes"], df["sgwNumberOfUes-sgwNbrOfUes"]
    )

    print(f"  -> {out.shape[1]} colonnes conservées / créées, {out.shape[0]} lignes")
    return out


# ---------------------------------------------------------------------------
# 3. PDC
# ---------------------------------------------------------------------------
def process_pdc(path):
    print("\n=== PDC ===")
    df = pd.read_csv(path)

    base_kpis = [
        "tot-run-time",
        "num-cmds-ok",
        "num-cmds-fail-timeout",
        "num-cmds-default-ok",
        "num-cmds-mapn-ok",
        "num-cmds-aapn-ok",
        "time-default",
        "time-mapn",
        "time-aapn",
        "login-time",
        "timeout-occured",
        "session-backout-fail",
    ]

    keep_cols =  ["Time"] + keep_existing(df, base_kpis)
    out = df[keep_cols].copy()

    total_commands = df["num-cmds-ok"] + df["num-cmds-fail-timeout"]
    out["total_commands"] = total_commands
    out["timeout_rate"] = safe_div(df["num-cmds-fail-timeout"], total_commands)
    out["command_success_rate"] = safe_div(df["num-cmds-ok"], total_commands)
    out["avg_runtime_per_command"] = safe_div(df["tot-run-time"], total_commands)
    out["avg_login_time"] = safe_div(df["login-time"], total_commands)
    out["configuration_time"] = df["time-default"] + df["time-mapn"] + df["time-aapn"]
    out["mapn_ratio"] = safe_div(df["num-cmds-mapn-ok"], total_commands)
    out["aapn_ratio"] = safe_div(df["num-cmds-aapn-ok"], total_commands)
    out["default_command_ratio"] = safe_div(df["num-cmds-default-ok"], total_commands)

    print(f"  -> {out.shape[1]} colonnes conservées / créées, {out.shape[0]} lignes")
    return out


# ---------------------------------------------------------------------------
# 4. STATISTICS
# ---------------------------------------------------------------------------
def process_statistics(path):
    print("\n=== STATISTICS ===")
    df = pd.read_csv(path)

    base_kpis = [
        "Active PDP contexts",
        "Active EPS bearers",
        "PDP creations",
        " Failed",
        "EPS bearer creations",
        " Failed.3",
        "PDP updates",
        " Failed.7",
        "EPS bearer modifications",
        " Failed.8",
        "PDP deactivations",
        " Failed.9",
        "Packets",
        "Bytes",
        "Dropped packets",
        "Packets.1",
        "Bytes.1",
        "Dropped packets.1",
        "Active subscribers",
        "Failed RADIUS Accounting procedures",
        "Total successful DT establishments",
        "Total requests for DT establishments",
        "DT RNC error indications",
    ]

    keep_cols = ["Time"] + keep_existing(df, base_kpis)
    out = df[keep_cols].copy()

    out["pdp_success_rate"] = safe_div(df["PDP creations"], df["PDP creations"] + df[" Failed"])
    out["pdp_failure_rate"] = safe_div(df[" Failed"], df["PDP creations"] + df[" Failed"])
    out["eps_success_rate"] = safe_div(df["EPS bearer creations"], df["EPS bearer creations"] + df[" Failed.3"])
    out["eps_failure_rate"] = safe_div(df[" Failed.3"], df["EPS bearer creations"] + df[" Failed.3"])
    out["update_failure_rate"] = safe_div(df[" Failed.7"], df["PDP updates"] + df[" Failed.7"])
    out["modification_failure_rate"] = safe_div(
        df[" Failed.8"], df["EPS bearer modifications"] + df[" Failed.8"]
    )
    out["deactivation_failure_rate"] = safe_div(
        df[" Failed.9"], df["PDP deactivations"] + df[" Failed.9"]
    )

    out["total_traffic"] = df["Bytes"] + df["Bytes.1"]
    out["total_packets"] = df["Packets"] + df["Packets.1"]
    out["total_dropped_packets"] = df["Dropped packets"] + df["Dropped packets.1"]
    out["packet_drop_rate"] = safe_div(out["total_dropped_packets"], out["total_packets"])
    out["avg_packet_size"] = safe_div(out["total_traffic"], out["total_packets"])
    out["traffic_per_subscriber"] = safe_div(out["total_traffic"], df["Active subscribers"])
    out["pdp_per_subscriber"] = safe_div(df["Active PDP contexts"], df["Active subscribers"])
    out["eps_per_subscriber"] = safe_div(df["Active EPS bearers"], df["Active subscribers"])

    out["dt_success_rate"] = safe_div(
        df["Total successful DT establishments"], df["Total requests for DT establishments"]
    )
    out["dt_error_rate"] = safe_div(
        df["DT RNC error indications"], df["Total requests for DT establishments"]
    )
    out["radius_failure_rate"] = safe_div(
        df["Failed RADIUS Accounting procedures"], df["Active subscribers"]
    )

    print(f"  -> {out.shape[1]} colonnes conservées / créées, {out.shape[0]} lignes")
    return out


# ---------------------------------------------------------------------------
# 5. PM Jobs EPG 
# ---------------------------------------------------------------------------
def process_pm_job(path):
    print("\n=== PM Jobs EPG ===")

    base_kpis = [
        "board-information:average-cpu-usage:",
        "board-information:peak-cpu-usage:",
        "board-information:memory:",
        "board-information:memory-used:",
        "ggsn-global-stats:",
        "ggsn-pdp-contexts-stats-attempted:",
        "ggsn-pdp-contexts-stats-completed:",
        "ggsn-pdp-contexts-stats-failed:",
        "ggsn-gtp-error-stats:",
        "ggsn-gtp-stats:",
        "ggsn-uplink-traffic-info:",
        "ggsn-downlink-traffic-info:",
        "ggsn-gtpu:",
    ]

    header = pd.read_csv(path, nrows=0).columns.tolist()
    usecols = [c for c in header if c == "Time" or any(c.startswith(p) for p in base_kpis)]    
    print(f"  Colonnes sélectionnées sur {len(header)} : {len(usecols)}")

    df = pd.read_csv(path, usecols=usecols, low_memory=False)

    def cols_starting(prefix):
        return [c for c in df.columns if c.startswith(prefix)]

    cpu_avg_cols = cols_starting("board-information:average-cpu-usage:")
    cpu_peak_cols = cols_starting("board-information:peak-cpu-usage:")
    mem_total_cols = cols_starting("board-information:memory:")
    mem_used_cols = cols_starting("board-information:memory-used:")

    
    out = pd.DataFrame()
    if cpu_avg_cols:
        out["cpu_mean"] = df[cpu_avg_cols].mean(axis=1)
    if cpu_peak_cols:
        out["cpu_max"] = df[cpu_peak_cols].max(axis=1)
    if mem_used_cols and mem_total_cols:
        out["memory_usage_rate"] = safe_div(
            df[mem_used_cols].sum(axis=1), df[mem_total_cols].sum(axis=1)
        )

    attempted_act = "ggsn-pdp-contexts-stats-attempted:ggsn-attempted-activation"
    completed_act = "ggsn-pdp-contexts-stats-completed:ggsn-completed-activation"
    failed_act = "ggsn-pdp-contexts-stats-failed:ggsn-failed-activation"

    if attempted_act in df.columns:
        if completed_act in df.columns:
            out["pdp_success_rate"] = safe_div(df[completed_act], df[attempted_act])
        if failed_act in df.columns:
            out["pdp_failure_rate"] = safe_div(df[failed_act], df[attempted_act])

    gtp_error_cols = cols_starting("ggsn-gtp-error-stats:")
    gtp_requests_col = "ggsn-gtp-stats:ggsn-gtp-requests-accepted"
    if gtp_error_cols and gtp_requests_col in df.columns:
        total_gtp_errors = df[gtp_error_cols].sum(axis=1)
        out["gtp_error_rate"] = safe_div(total_gtp_errors, df[gtp_requests_col])

    ul_bytes_col = "ggsn-uplink-traffic-info:ggsn-uplink-bytes"
    dl_bytes_col = "ggsn-downlink-traffic-info:ggsn-downlink-bytes"
    ul_packets_col = "ggsn-uplink-traffic-info:ggsn-uplink-packets"
    dl_packets_col = "ggsn-downlink-traffic-info:ggsn-downlink-packets"
    ul_drops_col = "ggsn-uplink-traffic-info:ggsn-uplink-drops"
    dl_drops_col = "ggsn-downlink-traffic-info:ggsn-downlink-drops"

    if ul_bytes_col in df.columns and dl_bytes_col in df.columns:
        out["total_bytes"] = df[ul_bytes_col] + df[dl_bytes_col]
    if ul_packets_col in df.columns and dl_packets_col in df.columns:
        out["total_packets"] = df[ul_packets_col] + df[dl_packets_col]
    if ul_drops_col in df.columns and dl_drops_col in df.columns and "total_packets" in out.columns:
        dropped = df[ul_drops_col] + df[dl_drops_col]
        out["drop_rate"] = safe_div(dropped, out["total_packets"])

    created_tunnels_col = "ggsn-gtp-stats:ggsn-gtp-nbr-of-created-tunnels"
    total_tunnels_col = "ggsn-gtp-stats:ggsn-gtp-nbr-of-tunnels"
    if created_tunnels_col in df.columns and total_tunnels_col in df.columns:
        out["tunnel_utilization"] = safe_div(df[created_tunnels_col], df[total_tunnels_col])

    active_pdp_col = "ggsn-global-stats:ggsn-nbr-of-active-pdp-contexts"
    subscribers_col = "ggsn-global-stats:ggsn-nbr-of-subscribers"
    if active_pdp_col in df.columns and subscribers_col in df.columns:
        out["pdp_per_subscriber"] = safe_div(df[active_pdp_col], df[subscribers_col])

    if "Time" in df.columns:
        out["Time"] = df["Time"].values
    elif df.index.name == "Time":
        out["Time"] = df.index
    else:
        raise KeyError("Colonne 'Time' introuvable ni en colonne ni en index de df")

    print(f"  -> {out.shape[1]} colonnes conservées / créées, {out.shape[0]} lignes")
    return out



def main():
    parser = argparse.ArgumentParser(description="Feature engineering réseau (PGW/SGW/PDC/STATISTICS/PM Jobs)")
    parser.add_argument("--input-dir", default=".", help="Dossier contenant les CSV *_cleaned.csv")
    parser.add_argument("--output-dir", default="./features_out", help="Dossier de sortie")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    jobs = [
        ("pgw_cleaned.csv", process_pgw, "pgw_features.csv"),
        ("sgw_cleaned.csv", process_sgw, "sgw_features.csv"),
        ("pdc_cleaned.csv", process_pdc, "pdc_features.csv"),
        ("statistics_cleaned.csv", process_statistics, "statistics_features.csv"),
        ("pm_job_epg-all_cleaned.csv", process_pm_job, "pm_job_features.csv"),
    ]

    for filename, func, out_name in jobs:
        in_path = os.path.join(args.input_dir, filename)
        if not os.path.exists(in_path):
            print(f"[!] Fichier introuvable, ignoré : {in_path}")
            continue
        out_df = func(in_path)
        out_path = os.path.join(args.output_dir, out_name)
        out_df.to_csv(out_path, index=False)
        print(f"  Sauvegardé -> {out_path}")

    print("\nTerminé.")


if __name__ == "__main__":
    main()

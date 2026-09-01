from __future__ import annotations
import numpy as np
import pandas as pd

TIME_COL_CANDIDATES = ["Time", "time", "timestamp"]


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def _safe_div(a, b):
    return _num(a) / _num(b).replace(0, np.nan)


def _find_time_col(df: pd.DataFrame) -> str | None:
    for c in TIME_COL_CANDIDATES:
        if c in df.columns:
            return c
    return None


def _nearest_to_grid(df: pd.DataFrame, grid: pd.DatetimeIndex, time_col: str) -> pd.DataFrame:
    x = df.copy()
    x["_ts"] = pd.to_datetime(x[time_col], errors="coerce")
    x = x.dropna(subset=["_ts"]).sort_values("_ts")
    q = pd.DataFrame({"_ts": grid})
    return pd.merge_asof(q, x, on="_ts", direction="nearest", tolerance=pd.Timedelta("7min"))


def _build_grid(raw_dfs: dict, warnings: list) -> pd.DatetimeIndex:
    all_ts = []
    for source, df in raw_dfs.items():
        if df is None:
            continue
        tcol = _find_time_col(df)
        if tcol is None:
            warnings.append(f"[{source}] Colonne de temps introuvable (Time/time/timestamp) — source ignorée.")
            continue
        ts = pd.to_datetime(df[tcol], errors="coerce").dropna()
        if len(ts):
            all_ts.append(ts)
    if not all_ts:
        raise ValueError("Aucune colonne de temps exploitable dans les fichiers uploadés.")
    combined = pd.concat(all_ts)
    start = combined.min().floor("15min")
    end = combined.max().ceil("15min")
    grid = pd.date_range(start=start, end=end, freq="15min")
    if len(grid) < 2:
        warnings.append("Grille temporelle très courte (< 2 points) — vérifiez l'étendue des données uploadées.")
    return grid


def build_engineered_features(raw_dfs: dict, warnings: list) -> pd.DataFrame:
    grid = _build_grid(raw_dfs, warnings)
    F = pd.DataFrame({"timestamp": grid})

    aligned = {}
    for source, df in raw_dfs.items():
        if df is None:
            warnings.append(f"[{source}] Fichier absent — colonnes {source}__* non générées (remplies à 0 plus loin).")
            continue
        tcol = _find_time_col(df)
        if tcol is None:
            continue
        aligned[source] = _nearest_to_grid(df, grid, tcol)

    # -------------------------------------------------------------------
    # PGW
    # -------------------------------------------------------------------
    if "pgw" in aligned:
        g = aligned["pgw"]
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
            if c in g.columns:
                F[f"pgw__{c}"] = _num(g[c])
            else:
                warnings.append(f"[pgw] Colonne absente : {c}")

        if "pdp-created__delta" in g.columns and "pdp-create-attempted__delta" in g.columns:
            F["pgw__pdp_success_rate"] = _safe_div(g["pdp-created__delta"], g["pdp-create-attempted__delta"])
        if "pdp-create-failed__delta" in g.columns and "pdp-create-attempted__delta" in g.columns:
            F["pgw__pdp_failure_rate"] = _safe_div(g["pdp-create-failed__delta"], g["pdp-create-attempted__delta"])
        if "eps-bearer-creation__delta" in g.columns and "eps-bearer-creation-attempted__delta" in g.columns:
            F["pgw__eps_success_rate"] = _safe_div(g["eps-bearer-creation__delta"], g["eps-bearer-creation-attempted__delta"])
        if "eps-bearer-creation-failed__delta" in g.columns and "eps-bearer-creation-attempted__delta" in g.columns:
            F["pgw__eps_failure_rate"] = _safe_div(g["eps-bearer-creation-failed__delta"], g["eps-bearer-creation-attempted__delta"])
        if "uplink_bytes__delta" in g.columns and "downlink_bytes__delta" in g.columns:
            F["pgw__total_traffic"] = _num(g["uplink_bytes__delta"]) + _num(g["downlink_bytes__delta"])
            F["pgw__ul_dl_traffic_ratio"] = _safe_div(g["uplink_bytes__delta"], g["downlink_bytes__delta"])
        if "uplink_packets__delta" in g.columns and "downlink_packets__delta" in g.columns:
            F["pgw__total_packets"] = _num(g["uplink_packets__delta"]) + _num(g["downlink_packets__delta"])
        if "uplink_dropped-packets__delta" in g.columns and "uplink_packets__delta" in g.columns:
            F["pgw__drop_rate_ul"] = _safe_div(g["uplink_dropped-packets__delta"], g["uplink_packets__delta"])
        if "downlink_dropped-packets__delta" in g.columns and "downlink_packets__delta" in g.columns:
            F["pgw__drop_rate_dl"] = _safe_div(g["downlink_dropped-packets__delta"], g["downlink_packets__delta"])
        if "pdn-suspended-connections" in g.columns and "pdn-pgw-connections" in g.columns:
            F["pgw__suspended_connection_ratio"] = _safe_div(g["pdn-suspended-connections"], g["pdn-pgw-connections"])

    # -------------------------------------------------------------------
    # SGW
    # -------------------------------------------------------------------
    if "sgw" in aligned:
        s = aligned["sgw"]
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
            if c in s.columns:
                F[f"sgw__{c}"] = _num(s[c])
            else:
                warnings.append(f"[sgw] Colonne absente : {c}")

        if all(c in s.columns for c in [
            "sgwGtpTunnelMgmtS4S11-smCreateSessionRespAccSent__delta",
            "sgwGtpTunnelMgmtS4S11-smCreateSessionReqRcvd__delta",
        ]):
            F["sgw__session_success_rate"] = _safe_div(
                s["sgwGtpTunnelMgmtS4S11-smCreateSessionRespAccSent__delta"],
                s["sgwGtpTunnelMgmtS4S11-smCreateSessionReqRcvd__delta"],
            )
        if all(c in s.columns for c in [
            "sgwGtpTunnelMgmtS4S11-smCreateSessionRespRejSent__delta",
            "sgwGtpTunnelMgmtS4S11-smCreateSessionReqRcvd__delta",
        ]):
            F["sgw__session_failure_rate"] = _safe_div(
                s["sgwGtpTunnelMgmtS4S11-smCreateSessionRespRejSent__delta"],
                s["sgwGtpTunnelMgmtS4S11-smCreateSessionReqRcvd__delta"],
            )
        if all(c in s.columns for c in [
            "sgwGtpTunnelMgmtS4S11-smModifyBearerRespRejSent__delta",
            "sgwGtpTunnelMgmtS4S11-smModifyBearerReqRcvd__delta",
        ]):
            F["sgw__bearer_modification_failure_rate"] = _safe_div(
                s["sgwGtpTunnelMgmtS4S11-smModifyBearerRespRejSent__delta"],
                s["sgwGtpTunnelMgmtS4S11-smModifyBearerReqRcvd__delta"],
            )
        if all(c in s.columns for c in ["sgwGtpTrafficS5S8-outDataByte", "sgwGtpTrafficS1uS4S12-outDataByte"]):
            F["sgw__total_traffic"] = (
                _num(s["sgwGtpTrafficS5S8-outDataByte"]) + _num(s["sgwGtpTrafficS1uS4S12-outDataByte"])
            )
        pkt_cols = [
            "sgwGtpTrafficS5S8-inDataPkt", "sgwGtpTrafficS5S8-outDataPkt",
            "sgwGtpTrafficS1uS4S12-inDataPkt", "sgwGtpTrafficS1uS4S12-outDataPkt",
        ]
        drop_cols = ["sgwUplinkTraffic-sgwUplinkDroppedPackets__delta", "sgwDownlinkTraffic-sgwDownlinkDroppedPackets__delta"]
        if all(c in s.columns for c in pkt_cols + drop_cols):
            F["sgw__packet_drop_rate"] = _safe_div(
                _num(s[drop_cols[0]]) + _num(s[drop_cols[1]]),
                _num(s[pkt_cols[0]]) + _num(s[pkt_cols[1]]) + _num(s[pkt_cols[2]]) + _num(s[pkt_cols[3]]),
            )
        if all(c in s.columns for c in ["sgwNumberOfUes-nbrOfSuspendedUes", "sgwNumberOfUes-sgwNbrOfUes"]):
            F["sgw__suspended_ue_ratio"] = _safe_div(s["sgwNumberOfUes-nbrOfSuspendedUes"], s["sgwNumberOfUes-sgwNbrOfUes"])
        if all(c in s.columns for c in ["sgwNumberOfSessions-sgwNbrOfIdleBearers", "sgwNumberOfSessions-sgwNbrOfBearers"]):
            F["sgw__idle_bearer_ratio"] = _safe_div(s["sgwNumberOfSessions-sgwNbrOfIdleBearers"], s["sgwNumberOfSessions-sgwNbrOfBearers"])
        if all(c in s.columns for c in ["sgwNumberOfUes-sgwNbrOfConnectedUes", "sgwNumberOfUes-sgwNbrOfUes"]):
            F["sgw__connected_ue_ratio"] = _safe_div(s["sgwNumberOfUes-sgwNbrOfConnectedUes"], s["sgwNumberOfUes-sgwNbrOfUes"])

    # -------------------------------------------------------------------
    # PDC
    # -------------------------------------------------------------------
    if "pdc" in aligned:
        d = aligned["pdc"]
        pdc_direct = [
            "tot-run-time", "num-cmds-ok", "num-cmds-fail-timeout",
            "num-cmds-default-ok", "num-cmds-mapn-ok", "num-cmds-aapn-ok",
            "time-default", "time-mapn", "time-aapn", "login-time",
            "timeout-occured", "session-backout-fail",
        ]
        for c in pdc_direct:
            if c in d.columns:
                F[f"pdc__{c}"] = _num(d[c])
            else:
                warnings.append(f"[pdc] Colonne absente : {c}")

        if "num-cmds-ok" in d.columns and "num-cmds-fail-timeout" in d.columns:
            cmd_total = _num(d["num-cmds-ok"]) + _num(d["num-cmds-fail-timeout"])
            F["pdc__timeout_rate"] = _safe_div(d["num-cmds-fail-timeout"], cmd_total)
            F["pdc__command_success_rate"] = _safe_div(d["num-cmds-ok"], cmd_total)
            if "tot-run-time" in d.columns:
                F["pdc__avg_runtime_per_command"] = _safe_div(d["tot-run-time"], cmd_total)
            if "num-cmds-mapn-ok" in d.columns:
                F["pdc__mapn_ratio"] = _safe_div(d["num-cmds-mapn-ok"], cmd_total)
            if "num-cmds-aapn-ok" in d.columns:
                F["pdc__aapn_ratio"] = _safe_div(d["num-cmds-aapn-ok"], cmd_total)
            if "num-cmds-default-ok" in d.columns:
                F["pdc__default_command_ratio"] = _safe_div(d["num-cmds-default-ok"], cmd_total)
        if "login-time" in d.columns:
            F["pdc__avg_login_time"] = _num(d["login-time"])  # valeur brute, cohérent avec build_data.py
        if all(c in d.columns for c in ["time-default", "time-mapn", "time-aapn"]):
            F["pdc__configuration_time"] = _num(d["time-default"]) + _num(d["time-mapn"]) + _num(d["time-aapn"])

    # -------------------------------------------------------------------
    # STATISTICS
    # -------------------------------------------------------------------
    if "stats" in aligned:
        st = aligned["stats"]
        stats_direct = [
            "Active PDP contexts", "Active EPS bearers", "PDP creations", " Failed",
            "EPS bearer creations", " Failed.3", "PDP updates", " Failed.7",
            "EPS bearer modifications", " Failed.8", "PDP deactivations", " Failed.9",
            "Packets", "Bytes", "Dropped packets", "Packets.1", "Bytes.1",
            "Dropped packets.1", "Active subscribers",
            "Failed RADIUS Accounting procedures",
            "Total successful DT establishments", "Total requests for DT establishments",
            "DT RNC error indications",
        ]
        for c in stats_direct:
            if c in st.columns:
                F[f"stats__{c}"] = _num(st[c])
            else:
                warnings.append(f"[stats] Colonne absente : {c}")

        if all(c in st.columns for c in ["PDP creations", " Failed"]):
            denom = _num(st["PDP creations"]) + _num(st[" Failed"])
            F["stats__pdp_success_rate"] = _safe_div(st["PDP creations"], denom)
            F["stats__pdp_failure_rate"] = _safe_div(st[" Failed"], denom)
        if all(c in st.columns for c in ["EPS bearer creations", " Failed.3"]):
            denom = _num(st["EPS bearer creations"]) + _num(st[" Failed.3"])
            F["stats__eps_success_rate"] = _safe_div(st["EPS bearer creations"], denom)
            F["stats__eps_failure_rate"] = _safe_div(st[" Failed.3"], denom)
        if all(c in st.columns for c in ["PDP updates", " Failed.7"]):
            F["stats__update_failure_rate"] = _safe_div(st[" Failed.7"], _num(st["PDP updates"]) + _num(st[" Failed.7"]))
        if all(c in st.columns for c in ["EPS bearer modifications", " Failed.8"]):
            F["stats__modification_failure_rate"] = _safe_div(st[" Failed.8"], _num(st["EPS bearer modifications"]) + _num(st[" Failed.8"]))
        if all(c in st.columns for c in ["PDP deactivations", " Failed.9"]):
            F["stats__deactivation_failure_rate"] = _safe_div(st[" Failed.9"], _num(st["PDP deactivations"]) + _num(st[" Failed.9"]))
        if all(c in st.columns for c in ["Bytes", "Bytes.1"]):
            F["stats__total_traffic"] = _num(st["Bytes"]) + _num(st["Bytes.1"])
        if all(c in st.columns for c in ["Packets", "Packets.1"]):
            F["stats__total_packets"] = _num(st["Packets"]) + _num(st["Packets.1"])
        if all(c in st.columns for c in ["Dropped packets", "Dropped packets.1"]):
            F["stats__total_dropped_packets"] = _num(st["Dropped packets"]) + _num(st["Dropped packets.1"])
        if all(c in st.columns for c in ["Dropped packets", "Dropped packets.1", "Packets", "Packets.1"]):
            F["stats__packet_drop_rate"] = _safe_div(
                _num(st["Dropped packets"]) + _num(st["Dropped packets.1"]),
                _num(st["Packets"]) + _num(st["Packets.1"]),
            )
            F["stats__avg_packet_size"] = _safe_div(
                _num(st.get("Bytes", 0)) + _num(st.get("Bytes.1", 0)),
                _num(st["Packets"]) + _num(st["Packets.1"]),
            )
        if all(c in st.columns for c in ["Bytes", "Bytes.1", "Active subscribers"]):
            F["stats__traffic_per_subscriber"] = _safe_div(_num(st["Bytes"]) + _num(st["Bytes.1"]), st["Active subscribers"])
        if all(c in st.columns for c in ["Active PDP contexts", "Active subscribers"]):
            F["stats__pdp_per_subscriber"] = _safe_div(st["Active PDP contexts"], st["Active subscribers"])
        if all(c in st.columns for c in ["Active EPS bearers", "Active subscribers"]):
            F["stats__eps_per_subscriber"] = _safe_div(st["Active EPS bearers"], st["Active subscribers"])
        if all(c in st.columns for c in ["Total successful DT establishments", "Total requests for DT establishments"]):
            F["stats__dt_success_rate"] = _safe_div(st["Total successful DT establishments"], st["Total requests for DT establishments"])
        if all(c in st.columns for c in ["DT RNC error indications", "Total requests for DT establishments"]):
            F["stats__dt_error_rate"] = _safe_div(st["DT RNC error indications"], st["Total requests for DT establishments"])
        if all(c in st.columns for c in ["Failed RADIUS Accounting procedures", "Active subscribers"]):
            F["stats__radius_failure_rate"] = _safe_div(st["Failed RADIUS Accounting procedures"], st["Active subscribers"])

    # -------------------------------------------------------------------
    # PM / EPG
    # -------------------------------------------------------------------
    if "pmjob" in aligned:
        pm0 = aligned["pmjob"]
        pm_num = pm0.drop(columns=["Time", "_ts"], errors="ignore").apply(pd.to_numeric, errors="coerce")
        cpu_avg = [c for c in pm_num.columns if "average-cpu-usage" in c]
        cpu_peak = [c for c in pm_num.columns if "peak-cpu-usage" in c]
        mem = [c for c in pm_num.columns if ":memory:" in c]
        mem_used = [c for c in pm_num.columns if ":memory-used:" in c]
        gtp_errors = [c for c in pm_num.columns if c.startswith("ggsn-gtp-error-stats:")]

        if cpu_avg:
            F["pmjob__cpu_mean"] = pm_num[cpu_avg].mean(axis=1)
        if cpu_peak:
            F["pmjob__cpu_max"] = pm_num[cpu_peak].max(axis=1)
        if mem_used and mem:
            F["pmjob__memory_usage_rate"] = _safe_div(pm_num[mem_used].sum(axis=1), pm_num[mem].sum(axis=1))

        attempted = "ggsn-pdp-contexts-stats-attempted:ggsn-attempted-activation"
        completed = "ggsn-pdp-contexts-stats-completed:ggsn-completed-activation"
        failed = "ggsn-pdp-contexts-stats-failed:ggsn-failed-activation"
        if attempted in pm_num.columns:
            if completed in pm_num.columns:
                F["pmjob__pdp_success_rate"] = _safe_div(pm_num[completed], pm_num[attempted])
            if failed in pm_num.columns:
                F["pmjob__pdp_failure_rate"] = _safe_div(pm_num[failed], pm_num[attempted])

        gtp_requests = "ggsn-gtp-stats:ggsn-gtp-requests-accepted"
        if gtp_errors and gtp_requests in pm_num.columns:
            F["pmjob__gtp_error_rate"] = _safe_div(pm_num[gtp_errors].sum(axis=1), pm_num[gtp_requests])

        ul_bytes, dl_bytes = "ggsn-uplink-traffic-info:ggsn-uplink-bytes", "ggsn-downlink-traffic-info:ggsn-downlink-bytes"
        ul_pkts, dl_pkts = "ggsn-uplink-traffic-info:ggsn-uplink-packets", "ggsn-downlink-traffic-info:ggsn-downlink-packets"
        ul_drops, dl_drops = "ggsn-uplink-traffic-info:ggsn-uplink-drops", "ggsn-downlink-traffic-info:ggsn-downlink-drops"
        if ul_bytes in pm_num.columns and dl_bytes in pm_num.columns:
            F["pmjob__total_bytes"] = pm_num[ul_bytes] + pm_num[dl_bytes]
        if ul_pkts in pm_num.columns and dl_pkts in pm_num.columns:
            F["pmjob__total_packets"] = pm_num[ul_pkts] + pm_num[dl_pkts]
        if ul_drops in pm_num.columns and dl_drops in pm_num.columns and ul_pkts in pm_num.columns and dl_pkts in pm_num.columns:
            F["pmjob__drop_rate"] = _safe_div(pm_num[ul_drops] + pm_num[dl_drops], pm_num[ul_pkts] + pm_num[dl_pkts])

        created_tun = "ggsn-gtp-stats:ggsn-gtp-nbr-of-created-tunnels"
        total_tun = "ggsn-gtp-stats:ggsn-gtp-nbr-of-tunnels"
        if created_tun in pm_num.columns and total_tun in pm_num.columns:
            F["pmjob__tunnel_utilization"] = _safe_div(pm_num[created_tun], pm_num[total_tun])

        active_pdp = "ggsn-global-stats:ggsn-nbr-of-active-pdp-contexts"
        subscribers = "ggsn-global-stats:ggsn-nbr-of-subscribers"
        if active_pdp in pm_num.columns and subscribers in pm_num.columns:
            F["pmjob__pdp_per_subscriber"] = _safe_div(pm_num[active_pdp], pm_num[subscribers])

    if not aligned:
        raise ValueError("Aucune source valide n'a pu être traitée (toutes les sources sont absentes ou illisibles).")

    return F

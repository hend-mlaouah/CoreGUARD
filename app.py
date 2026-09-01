import io
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import streamlit as st

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent))
from engine.load_models import (
    load_isolation_forest,
    load_scaler,
    load_xgboost_model,
)
from engine.pipeline import run_pipeline
from engine.rca import compute_shap_values, global_feature_importance, top_features_for_row
from engine.scoring import build_scored_dataset, get_model_feature_names, score_isolation_forest, score_xgboost
from ui.theme import ORANGE, inject_theme, orange_header, severity_badge_html
st.set_page_config(page_title=" CoreGuard | CORE-NETWORK", layout="wide")
inject_theme()

SOURCE_KEYWORDS = {
    "pgw": ["pgw"],
    "sgw": ["sgw"],
    "pdc": ["pdc"],
    "stats": ["stat"],
    "pmjob": ["pmjob", "pm_job", "pm job", "epg"],
}

SEVERITY_ORDER = {"Critique": 0, "Élevé": 1, "Moyen": 2, "Faible": 3}


st.markdown(
    """
    <style>
    .stepper { display:flex; align-items:flex-start; justify-content:center; margin: 0.5rem 0 2rem 0; }
    .step-item { display:flex; flex-direction:column; align-items:center; gap:0.45rem; width:120px; }
    .step-circle {
        width:38px; height:38px; border-radius:50%; display:flex; align-items:center;
        justify-content:center; font-weight:700; font-size:0.95rem;
        border:2px solid #444; color:#888; background:#161616;
    }
    .step-circle.active { border-color:#FF7900; color:#FF7900; background:#1F1F1F; box-shadow:0 0 0 4px rgba(255,121,0,0.14), 0 2px 8px rgba(255,121,0,0.25); }
    .step-circle.done { border-color:#4CAF50; color:#4CAF50; background:#1F1F1F; }
    .step-label { font-size:0.82rem; color:#777; }
    .step-label.active { color:#F5F5F5; font-weight:600; }
    .step-line { width:64px; height:2px; background:#444; margin-top:18px; }
    .step-line.done { background:#4CAF50; }
    .ds-card {
        border:1px solid #333; border-radius:10px; padding:1rem 1.2rem; margin-bottom:0.7rem;
        background:#161616;
    }
    .ds-card.valid { border-color:#FF7900; }
    .col-tag {
        display:inline-block; background:#2A1400; color:#FF7900; border:1px solid #5A3200;
        border-radius:14px; padding:2px 10px; margin:2px; font-size:0.78rem;
    }
    /* Masque uniquement la liste de navigation multipage Streamlit.
       La sidebar elle-même reste active (utilisée pour les filtres à l'étape Résultat). */
    [data-testid="stSidebarNav"] { display: none; }
    div[data-testid="stAppViewBlockContainer"] { max-width: 1100px; }
    .sidebar-section-title {
        text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.72rem;
        color: #888; font-weight: 700; margin: 0.2rem 0 0.8rem 0;
    }
    /* Barre de marque (logo Orange + nom de la plateforme) */
    .brand-bar {
        display:flex; align-items:center; justify-content:space-between;
        padding-bottom:1rem; margin-bottom:1.6rem;
        border-bottom: none;
        position: relative;
    }
    .brand-bar::after {
        content: "";
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, #FF7900 0%, #FFB577 50%, #FF7900 100%);
        box-shadow: 0 0 14px 1px rgba(255,121,0,0.55);
    }
    .brand-left { display:flex; align-items:center; gap:0.7rem; }
    .brand-mark {
        width:34px; height:34px; border-radius:8px; background:#FF7900;
        display:flex; align-items:center; justify-content:center;
        color:#0A0A0A; font-weight:800; font-size:0.72rem; letter-spacing:-0.02em;
        overflow:hidden;
    }
    .brand-mark img { width:100%; height:100%; object-fit:contain; }
    .brand-text b { display:block; color:#F5F5F5; font-size:1rem; line-height:1.2; }
    .brand-text span { display:block; color:#888; font-size:0.75rem; }
    .brand-badge {
        border:1px solid #3A3A3A; background:transparent; color:#888;
        font-size:0.66rem; font-weight:600; letter-spacing:0.05em;
        padding:5px 12px; border-radius:14px; white-space:nowrap;
    }
    .metric-card {
        border:1px solid #2E2E2E; border-radius:12px; padding:1.15rem 1.3rem;
        background:linear-gradient(165deg,#181818 0%,#0D0D0D 100%);
        position:relative; overflow:hidden; height:100%;
        box-shadow: 0 2px 10px rgba(0,0,0,0.35);
        transition: transform 0.15s ease-in-out, box-shadow 0.15s ease-in-out, border-color 0.15s ease-in-out;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 22px rgba(0,0,0,0.5);
        border-color:#444;
    }
    .metric-card .metric-label {
        color:#999; font-size:0.76rem; font-weight:600; margin-bottom:0.4rem;
        text-transform:uppercase; letter-spacing:0.06em;
    }
    .metric-card .metric-value {
        font-size:2.1rem; font-weight:800; color:#F5F5F5; line-height:1.1;
    }
    .metric-card.accent-orange { border-top:3px solid #FF7900; }
    .metric-card.accent-orange .metric-value { color:#FF7900; }
    .metric-card.accent-red { border-top:3px solid #E1000F; }
    .metric-card.accent-red .metric-value { color:#E1000F; }
    .metric-card.accent-neutral { border-top:3px solid #4A4A4A; }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_stepper(current_step: int):
    steps = [(1, "Dataset"), (2, "Configuration"), (3, "Résultat")]
    html = '<div class="stepper">'
    for i, (num, label) in enumerate(steps):
        if num < current_step:
            circle_class, content = "step-circle done", "✓"
        elif num == current_step:
            circle_class, content = "step-circle active", str(num)
        else:
            circle_class, content = "step-circle", str(num)
        label_class = "step-label active" if num <= current_step else "step-label"
        html += (
            f'<div class="step-item"><div class="{circle_class}">{content}</div>'
            f'<div class="{label_class}">{label}</div></div>'
        )
        if i < len(steps) - 1:
            line_class = "step-line done" if num < current_step else "step-line"
            html += f'<div class="{line_class}"></div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


if "wizard_step" not in st.session_state:
    st.session_state["wizard_step"] = 1


def go_to(step: int):
    st.session_state["wizard_step"] = step


def reset_wizard():
    for key in [
        "wizard_step", "raw_files", "scored_df", "model_input", "xgb_features",
        "pipeline_warnings", "used_local_scaler", "xgb_model", "shap_values",
        "shap_n_rows", "rca_pdf_bytes", "threshold",
    ]:
        st.session_state.pop(key, None)
    st.session_state["wizard_step"] = 1


import base64  

_APP_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = Path(__file__).resolve().parents[1]

_LOGO_CANDIDATES = [
    _PROJECT_ROOT / "assets" / f"orange_logo.{ext}" for ext in ("png", "jpg", "jpeg", "svg")
] + [
    _APP_DIR / "assets" / f"orange_logo.{ext}" for ext in ("png", "jpg", "jpeg", "svg")
] + [
    _PROJECT_ROOT / "assets" / f"logo.{ext}" for ext in ("png", "jpg", "jpeg", "svg")
] + [
    Path.cwd() / "assets" / f"orange_logo.{ext}" for ext in ("png", "jpg", "jpeg", "svg")
]

_LOGO_MIME = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "svg": "image/svg+xml"}


def _find_logo_path() -> Path | None:
    for candidate in _LOGO_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def _brand_mark_html() -> str:
    logo_path = _find_logo_path()
    if logo_path:
        ext = logo_path.suffix.lstrip(".").lower()
        mime = _LOGO_MIME.get(ext, "image/png")
        b64 = base64.b64encode(logo_path.read_bytes()).decode("utf-8")
        return f'<img src="data:{mime};base64,{b64}" alt="Orange" />'
    return "orange"



_HERO_BG_CANDIDATES = [
    _PROJECT_ROOT / "assets" / f"hero_bg.{ext}" for ext in ("jpg", "jpeg", "png", "webp")
] + [
    _APP_DIR / "assets" / f"hero_bg.{ext}" for ext in ("jpg", "jpeg", "png", "webp")
] + [
    _PROJECT_ROOT / "assets" / f"hero.{ext}" for ext in ("jpg", "jpeg", "png", "webp")
]
_HERO_BG_MIME = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}


def _find_hero_bg_data_uri() -> str | None:
    for candidate in _HERO_BG_CANDIDATES:
        if candidate.exists():
            ext = candidate.suffix.lstrip(".").lower()
            mime = _HERO_BG_MIME.get(ext, "image/jpeg")
            b64 = base64.b64encode(candidate.read_bytes()).decode("utf-8")
            return f"data:{mime};base64,{b64}"
    return None


st.markdown(
    f"""
    <div class="brand-bar">
        <div class="brand-left">
            <div class="brand-mark">{_brand_mark_html()}</div>
            <div class="brand-text">
                <b> CoreGuard </b>
                <span>Orange Tunisie </span>
            </div>
        </div>
        <div class="brand-badge">INNOVATION ORANGE DIGITAL CENTER </div>
    </div>
    """,
    unsafe_allow_html=True,
)
orange_header(
    "Détection d'anomalies et prévention d'incidents sur le core network ",
    "Détectez. Anticipez. Comprenez. — à partir de vos données réseau. ",
    bg_image_data_uri=_find_hero_bg_data_uri(),
)
render_stepper(st.session_state["wizard_step"])

# =============================================================================
# ÉTAPE 1 — Dataset
# =============================================================================
if st.session_state["wizard_step"] == 1:
    
    files = st.file_uploader(
        "Upload Data",
        type=["csv"],
        accept_multiple_files=True,
        key="upload_data",
    )

    def _guess_source(filename: str, used_sources: set) -> str | None:
        name = filename.lower()
        for source, keywords in SOURCE_KEYWORDS.items():
            if source in used_sources:
                continue
            if any(kw in name for kw in keywords):
                return source
        return None

    provided = {}
    unmatched = []
    if files:
        used = set()
        for f in files:
            source = _guess_source(f.name, used)
            if source:
                provided[source] = f
                used.add(source)
            else:
                unmatched.append(f)

    if unmatched:
        st.markdown("##### Association manuelle")
        st.caption("Certains fichiers n'ont pas été reconnus automatiquement — associez-les à une source.")
        for f in unmatched:
            available = [s for s in SOURCE_KEYWORDS if s not in provided]
            choice = st.selectbox(
                f"Source pour « {f.name} »",
                options=["(ignorer)"] + available,
                key=f"manual_source_{f.name}",
            )
            if choice != "(ignorer)":
                provided[choice] = f

    if provided:
        st.markdown("#### Fichiers chargés")
        for source, f in provided.items():
            try:
                f.seek(0)
                preview = pd.read_csv(f)
                f.seek(0)
                n_rows, n_cols = preview.shape
                col_names = list(preview.columns)
                tags_html = "".join(f'<span class="col-tag">{c}</span>' for c in col_names[:4])
                extra = f'<span class="col-tag">+{n_cols - 4}</span>' if n_cols > 4 else ""
                st.markdown(
                    f"""
                    <div class="ds-card valid">
                        <b>{f.name}</b><br>
                        <span style="color:#AAA;">{n_rows:,} lignes · {n_cols} colonnes</span>
                        &nbsp;&nbsp;{tags_html}{extra}
                    </div>
                    """.replace(",", " "),
                    unsafe_allow_html=True,
                )
            except Exception as exc:
                st.error(f"[{f.name}] Erreur de lecture : {exc}")
        st.success(f"{len(provided)} fichier(s) chargé(s) avec succès")

    st.session_state["raw_files"] = provided

    c1, c2 = st.columns([1, 1])
    with c1:
        st.button("Recommencer", on_click=reset_wizard, use_container_width=True)
    with c2:
        st.button(
            "Suivant : Configuration →",
            type="primary",
            use_container_width=True,
            disabled=not provided,
            on_click=go_to,
            args=(2,),
        )

# =============================================================================
# ÉTAPE 2 — Configuration
# =============================================================================
elif st.session_state["wizard_step"] == 2:
    st.subheader("Configuration de l'Analyse")

    threshold = st.slider(
        "Seuil de décision XGBoost (probabilité d'incident)",
        min_value=0.05, max_value=0.95,
        value=st.session_state.get("threshold", 0.5), step=0.05,
        help="Au-dessus de ce seuil, une ligne est classée comme incident probable.",
    )
    st.session_state["threshold"] = threshold

    c1, c2 = st.columns([1, 1])
    with c1:
        st.button("← Retour", use_container_width=True, on_click=go_to, args=(1,))
    with c2:
        launch = st.button("Lancer l'analyse", type="primary", use_container_width=True)

    if launch:
        provided = st.session_state.get("raw_files", {})
        if not provided:
            st.error("Aucun fichier chargé — retournez à l'étape Dataset.")
            st.stop()
        progress_box = st.container()

        def step_start(label: str):
            ph = progress_box.empty()
            ph.markdown(f"⏳ {label}")
            return ph

        def step_done(ph, label: str):
            ph.markdown(f" {label}")

        def step_failed(ph, label: str):
            ph.markdown(f"❌ {label}")

        ph = step_start("Chargement des modèles (Isolation Forest, XGBoost, scaler)...")
        try:
            iso_model = load_isolation_forest()
            xgb_model = load_xgboost_model()
            scaler = load_scaler()
        except Exception:
            step_failed(ph, "Chargement des modèles")
            raise
        step_done(ph, "Modèles chargés (Isolation Forest, XGBoost, scaler)")

        iso_features = get_model_feature_names(iso_model)
        xgb_features = get_model_feature_names(xgb_model)

        if not xgb_features:
            st.error(
                "Impossible de récupérer la liste des features attendues par le modèle XGBoost "
                "(`feature_names_in_` absent)."
            )
            st.stop()

        ph = step_start("Feature engineering + fenêtres glissantes + alignement...")
        try:
            result = run_pipeline(provided, iso_features, xgb_features, scaler=scaler)
        except Exception:
            step_failed(ph, "Feature engineering")
            raise
        step_done(ph, "Feature engineering terminé")

        if result.model_input.empty:
            st.error("Le pipeline n'a produit aucune ligne exploitable. Vérifiez le contenu des CSV uploadés.")
            for w in result.warnings:
                st.warning(w)
            st.stop()

        ph = step_start("Scoring Isolation Forest + XGBoost...")
        try:
            iso_result = score_isolation_forest(iso_model, result.model_input, iso_features)
            xgb_result = score_xgboost(xgb_model, result.model_input, xgb_features, threshold=threshold)
            scored = build_scored_dataset(result.model_input, iso_result, xgb_result)
        except Exception:
            step_failed(ph, "Scoring")
            raise
        step_done(ph, "Scoring terminé (Isolation Forest + XGBoost)")

        st.session_state["scored_df"] = scored
        st.session_state["model_input"] = result.model_input
        st.session_state["xgb_features"] = xgb_features
        st.session_state["pipeline_warnings"] = result.warnings
        st.session_state["used_local_scaler"] = result.used_local_scaler
        st.session_state["xgb_model"] = xgb_model
        st.session_state.pop("shap_values", None)
        st.session_state.pop("shap_n_rows", None)
        st.session_state.pop("rca_pdf_bytes", None)

        go_to(3)
        st.rerun()

# =============================================================================
# ÉTAPE 3 — Résultat (Dashboard + Explicabilité)
# =============================================================================
elif st.session_state["wizard_step"] == 3:
    if "scored_df" not in st.session_state:
        st.warning("Aucune analyse disponible — retournez à l'étape Dataset.")
        st.button("← Retour au Dataset", on_click=go_to, args=(1,))
        st.stop()

    scored = st.session_state["scored_df"]

    if st.session_state.get("used_local_scaler"):
        st.warning(
            "Aucun scaler d'entraînement (`models/scaler.pkl`) trouvé — une normalisation "
            "locale a été appliquée sur ce batch."
        )

    st.subheader("Résultat")

    def render_metric_card(label: str, value: str, accent: str = "neutral"):
        st.markdown(
            f"""
            <div class="metric-card accent-{accent}">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Lignes analysées", f"{len(scored):,}".replace(",", " "), "neutral")
    with c2:
        render_metric_card("Anomalies", str(int(scored["anomaly"].sum())), "orange")
    with c3:
        render_metric_card("Incidents probables", str(int(scored["incident_pred"].sum())), "orange")
    with c4:
        render_metric_card("Sévérité Critique", str(int((scored["severity"] == "Critique").sum())), "red")

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    # =========================================================================
    # Dashboard — Anomalies dans le temps (vue d'ensemble, pleine largeur)
    # =========================================================================
    st.markdown("#### Anomalies dans le temps")
    ts = scored.copy()
    ts.index = pd.to_datetime(ts.index, errors="coerce")
    daily = ts.groupby(ts.index.date)["anomaly"].sum()
    st.line_chart(daily)

    if "severity" in scored.columns:
        st.markdown("#### Répartition par sévérité")
        sev_order = ["Faible", "Moyen", "Élevé", "Critique"]
        counts = scored["severity"].value_counts().reindex(sev_order).fillna(0)
        st.bar_chart(counts)

    warnings = st.session_state.get("pipeline_warnings", [])
    if warnings:
        with st.expander(f"{len(warnings)} avertissement(s) pendant le pipeline"):
            for w in warnings:
                st.write("•", w)

    st.divider()

    # =========================================================================
    # Détail des anomalies — filtres dans la sidebar, résultats en pleine largeur
    # =========================================================================
    with st.sidebar:
        st.markdown('<div class="sidebar-section-title">Filtres</div>', unsafe_allow_html=True)

        sev_order2 = ["Critique", "Élevé", "Moyen", "Faible"]
        sev_filter = st.multiselect("Sévérité", sev_order2, default=sev_order2, key="anom_sev_filter")

        score_col = "incident_proba" if "incident_proba" in scored.columns else None
        if score_col:
            min_score, max_score = float(scored[score_col].min()), float(scored[score_col].max())
            seuil = st.slider(
                f"Seuil minimum ({score_col})",
                min_value=min_score, max_value=max_score, value=min_score,
                key="anom_seuil",
            )
        else:
            seuil = None

        kpi_cols = [c for c in scored.columns if c.startswith(("pdc__", "pgw__", "sgw__", "stats__", "pmjob__"))]
        kpi_bases = sorted({c.split("__roll_")[0].split("__lag_")[0] for c in kpi_cols})
        kpi_filter = st.multiselect(
            "Filtrer sur une KPI spécifiquee", kpi_bases, key="anom_kpi_filter"
        )

        def _reset_filters():
            st.session_state["anom_sev_filter"] = sev_order2
            st.session_state["anom_seuil"] = min_score if score_col else 0.0
            st.session_state["anom_kpi_filter"] = []

        st.button("Réinitialiser les filtres", use_container_width=True, on_click=_reset_filters)

        st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section-title">Légende sévérité</div>', unsafe_allow_html=True)
        for sev_label, sev_color in [
            ("Critique", "#AF1621"), ("Élevé","#FF7700"), ("Moyen", "#FFC107"), ("Faible", "#25A11A"),
        ]:
            st.markdown(
                f"""
                <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.35rem;">
                    <span style="width:10px;height:10px;border-radius:50%;background:{sev_color};display:inline-block;"></span>
                    <span style="font-size:0.82rem;color:#CCC;">{sev_label}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="border-top:1px solid #2A2A2A; padding-top:0.8rem;">
                <div style="font-size:0.75rem; color:#666; font-weight:600;">CoreGuard AI</div>
                <div style="font-size:0.7rem; color:#555;">Orange Digital Center · v1.0</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    filtered = scored.copy()
    if sev_filter and "severity" in filtered.columns:
        filtered = filtered[filtered["severity"].isin(sev_filter)]
    if score_col and seuil is not None:
        filtered = filtered[filtered[score_col] >= seuil]

    st.markdown("#### Détail des anomalies")
    st.caption(f"{len(filtered)} ligne(s) après filtrage sur {len(scored)} au total.")

    if "severity" in filtered.columns and len(filtered) > 0:
        display_cols = ["row_id"] if "row_id" in filtered.columns else []
        display_cols += [c for c in ["anomaly_score", "incident_proba", "severity"] if c in filtered.columns]
        if kpi_filter:
            display_cols += [c for c in filtered.columns if any(c.startswith(k) for k in kpi_filter)]

        st.markdown("##### Aperçu avec sévérité")
        preview = filtered[display_cols].head(30).reset_index()
        for _, prow in preview.iterrows():
            rcols = st.columns([2, 2, 2, 1.5, 3])
            rcols[0].write(str(prow.get("timestamp", prow.get("index", ""))))
            rcols[1].write(f"Score: {prow.get('anomaly_score', '—'):.3f}" if "anomaly_score" in prow else "—")
            rcols[2].write(f"Incident: {prow.get('incident_proba', 0):.1%}" if "incident_proba" in prow else "—")
            rcols[3].markdown(severity_badge_html(prow.get("severity", "—")), unsafe_allow_html=True)
            rcols[4].write("")
        st.divider()
    elif len(filtered) == 0:
        st.info("Aucune ligne ne correspond aux filtres sélectionnés.")
 
    if score_col:
        st.markdown("##### Distribution des scores")
        score_bins = pd.cut(filtered[score_col], bins=10)
        score_counts = score_bins.value_counts().sort_index()
        score_counts.index = score_counts.index.map(lambda iv: f"{iv.left:.2f}–{iv.right:.2f}")
        st.bar_chart(score_counts)

    st.download_button(
        "Télécharger la sélection en CSV",
        data=filtered.to_csv(index=True).encode("utf-8"),
        file_name="anomalies_filtrees.csv",
        mime="text/csv",
    )

    st.divider()

    # =========================================================================
    # Explicabilité (SHAP / RCA)
    # =========================================================================
    st.markdown("#### Explicabilité")

    model_input = st.session_state["model_input"]
    xgb_features = st.session_state["xgb_features"]
    xgb_model = st.session_state["xgb_model"]

    anomalies = scored[scored["anomaly"] == 1] if "anomaly" in scored.columns else scored
    if anomalies.empty:
        st.success("Aucune anomalie détectée sur ce run — rien à expliquer.")
    else:
        X_anomalies = model_input.loc[anomalies.index, xgb_features]

        if "shap_values" not in st.session_state or st.session_state.get("shap_n_rows") != len(X_anomalies):
            try:
                with st.spinner("Calcul des valeurs SHAP (TreeExplainer)..."):
                    shap_values, expected_value = compute_shap_values(xgb_model, X_anomalies)
                st.session_state["shap_values"] = shap_values
                st.session_state["shap_n_rows"] = len(X_anomalies)
            except ImportError:
                st.error("Le package `shap` n'est pas installé.")
                st.stop()
            except Exception as exc:
                st.error(f"Erreur pendant le calcul SHAP : {exc}")
                st.stop()

        shap_values = st.session_state["shap_values"]
        feature_names = list(X_anomalies.columns)

        st.markdown("##### Vue globale — facteurs de risque dominants")
        global_imp = global_feature_importance(shap_values, feature_names, top_n=15)

        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor("#0A0A0A")
        ax.set_facecolor("#0A0A0A")
        ax.barh(global_imp["feature"][::-1], global_imp["importance"][::-1], color=ORANGE)
        ax.tick_params(colors="white", labelsize=8)
        ax.set_xlabel("Importance moyenne (|SHAP|)", color="white")
        for spine in ax.spines.values():
            spine.set_color("#444444")
        st.pyplot(fig, use_container_width=True)

        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight", dpi=150)
        shap_summary_image_bytes = img_buffer.getvalue()
        plt.close(fig)

        with st.expander("Voir la table d'importance globale"):
            st.dataframe(global_imp, use_container_width=True)

        st.divider()
        st.markdown("##### Détail par anomalie")
        row_options = list(range(len(X_anomalies)))
        labels = [f"#{anomalies.iloc[i].get('row_id', i)} — {X_anomalies.index[i]}" for i in row_options]
        selected_pos = st.selectbox("Sélectionner une anomalie", row_options, format_func=lambda i: labels[i])

        row = anomalies.iloc[selected_pos]
        top_feats = top_features_for_row(shap_values, feature_names, selected_pos, top_n=8)

        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Probabilité d'incident", f"{row.get('incident_proba', 0):.1%}")
        rc2.markdown(f"**Sévérité**<br>{severity_badge_html(row.get('severity', '—'))}", unsafe_allow_html=True)
        rc3.metric("Score Isolation Forest", f"{row.get('anomaly_score', 0):.3f}")

        st.markdown("###### Facteurs contributifs (SHAP)")

        def _short_family(label: str) -> str:
            return str(label).split(" — ")[0].split(" - ")[0].strip()

        families_ordered = list(dict.fromkeys(_short_family(fam) for fam in top_feats["family"]))
        increasing = list(dict.fromkeys(
            _short_family(fam) for fam, val in zip(top_feats["family"], top_feats["shap_value"]) if val > 0
        ))
        decreasing = list(dict.fromkeys(
            _short_family(fam) for fam, val in zip(top_feats["family"], top_feats["shap_value"]) if val < 0
        ))

        summary = f"Cette anomalie s'explique surtout par des valeurs inhabituelles sur : **{', '.join(families_ordered)}**."
        if increasing:
            summary += f" Les signaux de **{', '.join(increasing)}** augmentent le risque prédit."
        if decreasing:
            summary += f" Les signaux de **{', '.join(decreasing)}** le diminuent."
        st.write(summary)

        with st.expander("Voir le détail des métriques"):
            for _, f in top_feats.iterrows():
                direction = "augmente" if f["shap_value"] > 0 else "diminue"
                st.markdown(f"**{f['feature']}** · _{f['family']}_ — {direction} le risque (SHAP = {f['shap_value']:.4f})")
                st.caption(f["explication"])

        st.divider()
        st.markdown("##### Rapport RCA")
        st.caption("Génère un PDF récapitulatif de toutes les anomalies détectées, avec les facteurs SHAP et la sévérité.")

        if st.button("Générer le rapport RCA (PDF)", type="primary"):
            try:
                from engine.pdf_report import generate_rca_pdf

                top_features_by_row = {}
                for i in range(len(X_anomalies)):
                    rid = anomalies.iloc[i].get("row_id", i)
                    top_features_by_row[rid] = top_features_for_row(shap_values, feature_names, i, top_n=5)

                pdf_bytes = generate_rca_pdf(
                    anomalies_df=anomalies,
                    top_features_by_row=top_features_by_row,
                    global_importance_df=global_imp,
                    shap_summary_image_bytes=shap_summary_image_bytes,
                )
                st.session_state["rca_pdf_bytes"] = pdf_bytes
                st.success("Rapport généré.")
            except Exception as exc:
                st.error(f"Erreur pendant la génération du PDF : {exc}")

        if "rca_pdf_bytes" in st.session_state:
            st.download_button(
                "Télécharger le rapport RCA (PDF)",
                data=st.session_state["rca_pdf_bytes"],
                file_name="rapport_rca_aiops.pdf",
                mime="application/pdf",
            )

    st.divider()
    cB1, cB2 = st.columns([1, 1])
    with cB1:
        st.button("← Modifier la configuration", use_container_width=True, on_click=go_to, args=(2,))
    with cB2:
        st.button("Nouvelle analyse", use_container_width=True, on_click=reset_wizard)
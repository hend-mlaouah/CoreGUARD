from __future__ import annotations
import io
from datetime import datetime
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ORANGE = colors.HexColor("#FF7900")
ORANGE_DARK = colors.HexColor("#CC5F00")
BLACK = colors.HexColor("#0A0A0A")
GREY = colors.HexColor("#F2F2F2")
GREY_TEXT = colors.HexColor("#666666")
WHITE = colors.white

SEVERITY_COLORS = {
    "Faible": colors.HexColor("#4CAF50"),
    "Moyen": colors.HexColor("#FFC107"),
    "Élevé": colors.HexColor("#FF7900"),
    "Critique": colors.HexColor("#E1000F"),
}

SEVERITY_ORDER = {"Critique": 0, "Élevé": 1, "Moyen": 2, "Faible": 3}

MAX_ANOMALIES_IN_TABLE = 15

MARGIN = 1.6 * cm
PAGE_WIDTH, _ = A4
CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN


def _styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="HeaderTitle", fontSize=18, textColor=ORANGE, fontName="Helvetica-Bold", leading=22))
    styles.add(ParagraphStyle(name="HeaderSubtitle", fontSize=9.5, textColor=colors.HexColor("#BBBBBB"), leading=13))
    styles.add(ParagraphStyle(name="SectionBar", fontSize=9.5, textColor=BLACK, fontName="Helvetica-Bold", leading=12))
    styles.add(ParagraphStyle(name="Body", fontSize=9.5, textColor=BLACK, leading=14))
    styles.add(ParagraphStyle(name="AlertText", fontSize=10, textColor=WHITE, fontName="Helvetica-Bold", leading=13))
    styles.add(ParagraphStyle(name="Cell", fontSize=7.5, textColor=BLACK, leading=10))
    styles.add(ParagraphStyle(name="CellBold", fontSize=7.5, textColor=BLACK, leading=10, fontName="Helvetica-Bold"))
    return styles


def _header_banner(title: str, generated_at: str, n_anomalies: int, styles) -> Table:
    t = Table(
        [[Paragraph(title, styles["HeaderTitle"])],
         [Paragraph(f"Généré le {generated_at} &nbsp;·&nbsp; {n_anomalies} anomalie(s) analysée(s)", styles["HeaderSubtitle"])]],
        colWidths=[CONTENT_WIDTH],
    )
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BLACK),
        ("TOPPADDING", (0, 0), (-1, 0), 12),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 2),
        ("TOPPADDING", (0, 1), (-1, 1), 2),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 12),
        ("LEFTPADDING", (0, 0), (-1, -1), 16),
        ("RIGHTPADDING", (0, 0), (-1, -1), 16),
        ("LINEBELOW", (0, -1), (-1, -1), 2, ORANGE),
    ]))
    return t


def _section_bar(label: str, styles) -> Table:
    t = Table([[Paragraph(label.upper(), styles["SectionBar"])]], colWidths=[CONTENT_WIDTH])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), GREY),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, -1), 1.2, ORANGE),
    ]))
    return t


def _alert_banner(critique_count: int, styles) -> Table:
    text = (
        f"{critique_count} anomalie(s) de sévérité Critique détectée(s) — "
        f"action prioritaire recommandée."
    )
    t = Table([[Paragraph(text, styles["AlertText"])]], colWidths=[CONTENT_WIDTH])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), SEVERITY_COLORS["Critique"]),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ]))
    return t


def _severity_cell(sev: str, styles) -> Paragraph:
    color = SEVERITY_COLORS.get(sev, colors.grey)
    return Paragraph(f"<font color='{color.hexval()}'><b>{sev}</b></font>", styles["Cell"])


def generate_rca_pdf(
    anomalies_df: pd.DataFrame,
    top_features_by_row: dict,
    global_importance_df: pd.DataFrame | None = None,
    shap_summary_image_bytes: bytes | None = None,
    title: str = "Rapport RCA — CoreGuard",
    max_anomalies: int = MAX_ANOMALIES_IN_TABLE,
) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        topMargin=1.2 * cm, bottomMargin=1.6 * cm, leftMargin=MARGIN, rightMargin=MARGIN,
    )
    styles = _styles()
    story = []

    sev_counts = anomalies_df["severity"].value_counts()
    critique_count = int(sev_counts.get("Critique", 0))

    story.append(_header_banner(title, datetime.now().strftime("%d/%m/%Y à %H:%M"), len(anomalies_df), styles))
    story.append(Spacer(1, 14))

    if critique_count > 0:
        story.append(_alert_banner(critique_count, styles))
        story.append(Spacer(1, 14))

    story.append(_section_bar("Résumé exécutif", styles))
    story.append(Spacer(1, 8))
    summary_data = [["Sévérité", "Nombre"]] + [
        [s, str(int(sev_counts.get(s, 0)))] for s in ["Critique", "Élevé", "Moyen", "Faible"]
    ]
    t = Table(summary_data, colWidths=[CONTENT_WIDTH * 0.6, CONTENT_WIDTH * 0.4])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BLACK),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#DDDDDD")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, GREY]),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(t)
    story.append(Spacer(1, 16))

    if global_importance_df is not None and not global_importance_df.empty:
        story.append(_section_bar("Facteurs de risque globaux (SHAP)", styles))
        story.append(Spacer(1, 8))
        if shap_summary_image_bytes:
            story.append(Image(io.BytesIO(shap_summary_image_bytes), width=CONTENT_WIDTH, height=CONTENT_WIDTH * 0.45))
            story.append(Spacer(1, 10))
        top6 = global_importance_df.head(6)
        gi_data = [["Feature", "Famille", "Importance"]]
        for _, r in top6.iterrows():
            gi_data.append([
                Paragraph(str(r["feature"]), styles["Cell"]),
                Paragraph(str(r["family"]), styles["Cell"]),
                f"{r['importance']:.3f}",
            ])
        t2 = Table(gi_data, colWidths=[CONTENT_WIDTH * 0.36, CONTENT_WIDTH * 0.46, CONTENT_WIDTH * 0.18], repeatRows=1)
        t2.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8E8E8")),
            ("TEXTCOLOR", (0, 0), (-1, 0), BLACK),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#DDDDDD")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, GREY]),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(t2)
        story.append(Spacer(1, 16))

    story.append(_section_bar(
        f"Anomalies prioritaires (top {min(max_anomalies, len(anomalies_df))} sur {len(anomalies_df)})",
        styles,
    ))
    story.append(Spacer(1, 8))

    ranked = anomalies_df.assign(
        _sev_rank=anomalies_df["severity"].map(SEVERITY_ORDER).fillna(99),
    ).sort_values(
        by=["_sev_rank", "incident_proba"] if "incident_proba" in anomalies_df.columns else ["_sev_rank"],
        ascending=[True, False] if "incident_proba" in anomalies_df.columns else [True],
    ).head(max_anomalies)

    header_row = [
        Paragraph("ID", styles["CellBold"]),
        Paragraph("Horodatage", styles["CellBold"]),
        Paragraph("Sévérité", styles["CellBold"]),
        Paragraph("Proba.", styles["CellBold"]),
        Paragraph("Facteur dominant (SHAP)", styles["CellBold"]),
    ]
    rows = [header_row]
    for row_id, row in ranked.iterrows():
        rid = row.get("row_id", row_id)
        sev = row.get("severity", "—")
        proba = row.get("incident_proba", 0.0)
        ts = str(row.name)

        feats = top_features_by_row.get(rid)
        if feats is not None and not feats.empty:
            f0 = feats.iloc[0]
            direction = "▲" if f0["shap_value"] > 0 else "▼"
            factor_text = f"{direction} <b>{f0['feature']}</b> — {f0['explication']}"
        else:
            factor_text = "—"

        rows.append([
            Paragraph(str(rid), styles["Cell"]),
            Paragraph(ts, styles["Cell"]),
            _severity_cell(sev, styles),
            Paragraph(f"{proba:.1%}", styles["Cell"]),
            Paragraph(factor_text, styles["Cell"]),
        ])

    t3 = Table(
        rows,
        colWidths=[CONTENT_WIDTH * 0.07, CONTENT_WIDTH * 0.17, CONTENT_WIDTH * 0.11, CONTENT_WIDTH * 0.09, CONTENT_WIDTH * 0.56],
        repeatRows=1,
    )
    t3.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8E8E8")),
        ("TEXTCOLOR", (0, 0), (-1, 0), BLACK),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#DDDDDD")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, GREY]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(t3)

    if len(anomalies_df) > max_anomalies:
        story.append(Spacer(1, 8))
        story.append(Paragraph(
            f"+ {len(anomalies_df) - max_anomalies} autre(s) anomalie(s) détectée(s) — "
            f"voir l'export CSV complet dans l'onglet Anomalies de l'application pour le détail.",
            styles["HeaderSubtitle"],
        ))

    doc.build(story)
    return buffer.getvalue()

"""
Thème visuel noir / orange (charte Orange Entreprise).

Injecté via st.markdown(unsafe_allow_html=True) en tête de chaque page.
Couleur de marque officielle Orange : #FF7900.
"""

import streamlit as st

ORANGE = "#FF7900"
ORANGE_DARK = "#CC5F00"
ORANGE_LIGHT = "#FF9433"
BLACK = "#0A0A0A"
GREY_DARK = "#161616"
GREY_MID = "#1F1F1F"
WHITE = "#F5F5F5"

SEVERITY_COLORS = {
    "Faible": "#4CAF50",
    "Moyen": "#FFC107",
    "Élevé": "#FF7900",
    "Critique": "#E1000F",
}


def inject_theme():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, .stApp, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}

        .stApp {{
            background-color: {BLACK};
            background-image:
                radial-gradient(ellipse 900px 500px at 50% -10%, rgba(204,95,0,0.06), transparent 60%),
                radial-gradient(ellipse 700px 400px at 90% 10%, rgba(204,95,0,0.03), transparent 60%);
            background-attachment: fixed;
            color: {WHITE};
        }}

        section[data-testid="stSidebar"] {{
            background-color: {GREY_DARK};
            border-right: 1px solid {ORANGE_DARK};
        }}

        h1, h2, h3, h4 {{
            color: {WHITE} !important;
            letter-spacing: -0.01em;
        }}

        h1 {{
            border-bottom: 3px solid {ORANGE};
            padding-bottom: 0.4rem;
        }}

        /* Titres de section avec accent orange */
        .orange-accent {{
            color: {ORANGE} !important;
        }}

        /* --------------------------------------------------------------
           Hero (bloc titre principal, centré) — utilisé par orange_header()
        -------------------------------------------------------------- */
        .hero-block {{
            text-align: center;
            max-width: 780px;
            margin: 0.6rem auto 2.4rem auto;
            padding: 0 1rem;
        }}
        .hero-block h1 {{
            border-bottom: none !important;
            padding-bottom: 0 !important;
            font-size: 2.05rem;
            font-weight: 800;
            line-height: 1.28;
            margin-bottom: 0.9rem;
        }}
        .hero-divider {{
            width: 64px;
            height: 3px;
            background: linear-gradient(90deg, {ORANGE_DARK}, {ORANGE}, {ORANGE_LIGHT});
            margin: 0 auto 1.15rem auto;
            border-radius: 2px;
        }}
        .hero-subtitle {{
            color: #AAAAAA;
            font-size: 1rem;
            font-weight: 400;
            line-height: 1.55;
            margin: 0 auto;
        }}

        /* --------------------------------------------------------------
           Hero avec image de fond (variante utilisée si une image
           assets/hero_bg.* est trouvée) — bannière pleine largeur avec
           overlay sombre pour garder le texte lisible.
        -------------------------------------------------------------- */
        .hero-banner {{
            position: relative;
            border-radius: 14px;
            overflow: hidden;
            margin: 0.6rem 0 2.4rem 0;
            min-height: 260px;
            display: flex;
            align-items: center;
            justify-content: center;
            background-size: cover;
            background-position: center;
            padding: 3rem 1.5rem;
        }}
        .hero-banner::before {{
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(180deg, rgba(10,10,10,0.55) 0%, rgba(10,10,10,0.88) 100%);
            z-index: 1;
        }}
        .hero-banner-content {{
            position: relative;
            z-index: 2;
            text-align: center;
            max-width: 780px;
        }}
        .hero-banner-content h1 {{
            border-bottom: none !important;
            padding-bottom: 0 !important;
            font-size: 2.05rem;
            font-weight: 800;
            line-height: 1.28;
            margin-bottom: 0.9rem;
            color: {WHITE} !important;
        }}
        .hero-banner-content .hero-subtitle {{
            color: #DDDDDD;
        }}

        /* Boutons */
        div.stButton > button, div.stDownloadButton > button {{
            background: linear-gradient(135deg, {ORANGE} 0%, {ORANGE_DARK} 100%);
            color: {BLACK};
            border: none;
            font-weight: 600;
            border-radius: 6px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.4);
            transition: box-shadow 0.15s ease-in-out, transform 0.15s ease-in-out;
        }}
        div.stButton > button:hover, div.stDownloadButton > button:hover {{
            box-shadow: 0 4px 14px rgba(255,121,0,0.35);
            transform: translateY(-1px);
            color: {BLACK};
        }}

        /* Metrics */
        div[data-testid="stMetric"] {{
            background-color: {GREY_DARK};
            border: 1px solid #333;
            border-left: 4px solid {ORANGE};
            border-radius: 8px;
            padding: 0.8rem 1rem;
        }}
        div[data-testid="stMetricValue"] {{
            color: {ORANGE} !important;
        }}

        /* Onglets */
        button[data-baseweb="tab"] {{
            color: {WHITE};
            font-weight: 500;
        }}
        button[data-baseweb="tab"][aria-selected="true"] {{
            color: {ORANGE} !important;
            border-bottom-color: {ORANGE} !important;
            font-weight: 600;
        }}
        div[data-baseweb="tab-highlight"] {{
            background-color: {ORANGE} !important;
        }}

        /* Tables / dataframes */
        div[data-testid="stDataFrame"] {{
            border: 1px solid #333;
            border-radius: 6px;
        }}

        /* Uploader */
        section[data-testid="stFileUploaderDropzone"] {{
            background-color: {GREY_MID};
            border: 2px dashed {ORANGE};
            border-radius: 8px;
            transition: border-color 0.15s ease-in-out, background-color 0.15s ease-in-out;
        }}
        section[data-testid="stFileUploaderDropzone"]:hover {{
            border-color: {ORANGE_LIGHT};
            background-color: #232323;
        }}

        /* Alertes */
        div[data-baseweb="notification"] {{
            border-left: 4px solid {ORANGE};
        }}

        /* Badges de sévérité custom (utilisés via st.markdown) */
        .severity-badge {{
            display: inline-block;
            padding: 2px 10px;
            border-radius: 12px;
            font-weight: 600;
            font-size: 0.85rem;
            color: {BLACK};
        }}

        /* Progress bar */
        div[data-testid="stProgress"] > div > div {{
            background-color: {ORANGE};
        }}

        /* Liens */
        a {{ color: {ORANGE}; }}

        /* Séparateurs */
        hr {{ border-color: #333; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def severity_badge_html(severity: str) -> str:
    color = SEVERITY_COLORS.get(severity, "#888888")
    return f'<span class="severity-badge" style="background-color:{color};">{severity}</span>'


def orange_header(title: str, subtitle: str | None = None, bg_image_data_uri: str | None = None):
    """
    Affiche le bloc titre principal, centré.

    Si `bg_image_data_uri` est fourni (ex: "data:image/jpeg;base64,...."),
    le titre est affiché en bannière pleine largeur avec cette image en
    fond et un overlay sombre pour la lisibilité (comme une hero image de
    site vitrine). Sinon, fallback sur le bloc titre simple sans image.
    """
    subtitle_html = f'<p class="hero-subtitle">{subtitle}</p>' if subtitle else ""

    if bg_image_data_uri:
        st.markdown(
            f"""
            <div class="hero-banner" style="background-image:url('{bg_image_data_uri}');">
                <div class="hero-banner-content">
                    <h1>{title}</h1>
                    <div class="hero-divider"></div>
                    {subtitle_html}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="hero-block">
                <h1>{title}</h1>
                <div class="hero-divider"></div>
                {subtitle_html}
            </div>
            """,
            unsafe_allow_html=True,
        )


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import pickle
import lime
import lime.lime_tabular
import warnings
warnings.filterwarnings('ignore')

# ---------- NEW IMPORTS ----------
from job_role_matcher import suggest_roles
from project_analyzer import analyze_project
from recommendation_engine import generate_recommendations
# --------------------------------

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Placement AI Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# DESIGN TOKENS (single source of truth for charts)
# -----------------------------------------------------
# Page #F5F7FB   Surface #FFFFFF   Sidebar ink #101828   Border #E4E7EC
# Text #101828 / #344054 / muted #667085
# Accent #4C5FD5 (blue-indigo)   Success #10B981   Warning #F59E0B   Danger #E5484D
# Chart family: indigo, sky, teal, violet, then amber / red / slate for semantics
# =====================================================
PRIMARY = "#4C5FD5"
PALETTE = ["#4C5FD5", "#0EA5E9", "#14B8A6", "#8B7CF6", "#F59E0B", "#E5484D", "#94A3B8"]
FONT_STACK = "'Plus Jakarta Sans', Inter, system-ui, -apple-system, 'Segoe UI', sans-serif"

# =====================================================
# GLOBAL CSS
# Every rule below has a job: force a light, high-contrast surface so that
# text is readable whatever the visitor's OS/Streamlit theme is.
# =====================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

/* =====================================================
   TOKENS
   Page #F5F7FB | Surface #FFFFFF | Sidebar ink #101828
   Accent: blue-indigo #4C5FD5 (white text on it = 5.4:1)
   Text #101828 / #344054 / muted #667085
   ===================================================== */
:root {
    color-scheme: light;
    --bg: #F5F7FB;
    --surface: #FFFFFF;
    --surface-2: #EEF1F7;
    --border: #E4E7EC;
    --border-strong: #D0D5DD;
    --text: #101828;
    --text-2: #344054;
    --muted: #667085;
    --primary: #4C5FD5;
    --primary-dark: #3A4BC0;
    --primary-hover: #2F3EA8;
    --primary-soft: #EEF1FD;
    --primary-line: #C5CCF5;
    --primary-ink: #2B3790;
    --success: #10B981;
    --success-ink: #047857;
    --warning: #F59E0B;
    --warning-ink: #B54708;
    --danger: #E5484D;
    --danger-ink: #B42318;
}

/* Sidebar is dark: re-point the text tokens so every shared class adapts */
section[data-testid="stSidebar"] {
    --text: #F2F4F7;
    --text-2: #D0D5DD;
    --muted: #98A2B3;
    --border: rgba(255, 255, 255, 0.10);
    --border-strong: rgba(255, 255, 255, 0.18);
}
/* ...but form controls inside the sidebar stay light, so restore dark text there */
section[data-testid="stSidebar"] :where(div[data-baseweb="input"], div[data-baseweb="select"], div[data-baseweb="textarea"]) {
    --text: #101828;
    --text-2: #344054;
    --muted: #667085;
    --border-strong: #D0D5DD;
}

/* =====================================================
   FONT: Plus Jakarta Sans everywhere (icons + code excepted below)
   ===================================================== */
html, body, .stApp, .stApp *,
div[data-baseweb="popover"], div[data-baseweb="popover"] *,
ul[role="listbox"], ul[role="listbox"] * {
    font-family: 'Plus Jakarta Sans', system-ui, -apple-system, 'Segoe UI', sans-serif !important;
}
.stApp [data-testid="stIconMaterial"],
.stApp .material-icons,
.stApp .material-symbols-rounded,
.stApp [class*="material-symbols"] {
    font-family: 'Material Symbols Rounded', 'Material Icons' !important;
}
.stApp code, .stApp pre, .stApp pre * {
    font-family: ui-monospace, 'SF Mono', Menlo, Consolas, monospace !important;
}

/* =====================================================
   BASE SURFACE & LAYOUT (compact)
   ===================================================== */
html, body, .stApp {
    background: var(--bg) !important;
    color: var(--text-2);
    -webkit-font-smoothing: antialiased;
}
.block-container {
    padding: 1.75rem 1.75rem 2rem 1.75rem !important;
    max-width: 1200px !important;
}
.stApp [data-testid="stVerticalBlock"] { gap: 0.65rem; }
footer, #MainMenu { visibility: hidden; }
.stDeployButton, [data-testid="stDeployButton"] { display: none !important; }

/* Transparent header keeps the sidebar toggle usable */
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stHeader"] button,
[data-testid="stToolbar"] button,
[data-testid="stExpandSidebarButton"],
[data-testid="collapsedControl"] button { color: var(--text-2) !important; }
[data-testid="stSidebarCollapseButton"] button { color: #D0D5DD !important; }

/* =====================================================
   TYPOGRAPHY
   H1 28 / section 19 / card 16 / body 15 / caption 13
   ===================================================== */
[data-testid="stMarkdownContainer"] { color: var(--text-2) !important; }
:where(.stApp) :where([data-testid="stMarkdownContainer"]) :where(p, li) {
    color: var(--text-2) !important;
    font-size: 0.9375rem;
    line-height: 1.55;
}
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
    color: var(--text) !important;
    letter-spacing: -0.02em;
    font-weight: 700 !important;
}
.stApp h1 { font-size: 1.75rem !important; }
.stApp h2 { font-size: 1.3rem !important; padding: 0.2rem 0 0.1rem 0 !important; }
.stApp h3 { font-size: 1.1rem !important; padding: 0.2rem 0 0.1rem 0 !important; }
.stApp h4 { font-size: 1rem !important; padding: 0.15rem 0 0.1rem 0 !important; }
[data-testid="stMarkdownContainer"] strong { color: var(--text) !important; font-weight: 700; }
[data-testid="stMarkdownContainer"] a { color: var(--primary-dark) !important; }
[data-testid="stMarkdownContainer"] code {
    background: var(--primary-soft) !important;
    color: var(--primary-ink) !important;
    border-radius: 4px;
    padding: 1px 6px;
}
[data-testid="stHeaderActionElements"] { display: none !important; }
hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 1rem 0 !important; }
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
    color: var(--muted) !important;
    font-size: 0.8125rem !important;
}
[data-testid="stSpinner"] p, [data-testid="stSpinner"] div { color: var(--muted) !important; }

/* =====================================================
   SIDEBAR (dark ink, clearly distinct from the page)
   ===================================================== */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div,
[data-testid="stSidebarContent"] {
    background: #101828 !important;
}
section[data-testid="stSidebar"] { border-right: none !important; }
[data-testid="stSidebarUserContent"] { padding-top: 0.5rem; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { color: var(--text-2) !important; }

/* Navigation: radio becomes a vertical menu */
[data-testid="stSidebar"] div[role="radiogroup"] { gap: 2px; }
[data-testid="stSidebar"] div[role="radiogroup"] > label {
    display: flex; width: 100%; margin: 0;
    padding: 8px 12px;
    border-radius: 8px;
    border: 1px solid transparent;
    background: transparent;
    cursor: pointer;
    transition: background 0.15s ease, border-color 0.15s ease;
}
[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child { display: none !important; }
[data-testid="stSidebar"] div[role="radiogroup"] > label p {
    color: #D0D5DD !important;
    font-weight: 500;
    font-size: 0.9rem !important;
    margin: 0;
}
[data-testid="stSidebar"] div[role="radiogroup"] > label:hover { background: rgba(255, 255, 255, 0.07); }
[data-testid="stSidebar"] div[role="radiogroup"] > label:hover p { color: #FFFFFF !important; }
[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked),
[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"] {
    background: rgba(76, 95, 213, 0.30);
    border-color: rgba(165, 180, 252, 0.40);
}
[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p,
[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"] p {
    color: #FFFFFF !important;
    font-weight: 700;
}

.brand { display: flex; align-items: center; gap: 10px; margin-bottom: 2px; }
.brand-mark {
    width: 32px; height: 32px; border-radius: 8px;
    background: linear-gradient(135deg, #6B7BEA, #4C5FD5);
    color: #FFFFFF; font-weight: 800; font-size: 16px;
    display: flex; align-items: center; justify-content: center;
}
.brand-name { font-size: 1.05rem; font-weight: 700; color: var(--text); line-height: 1.2; letter-spacing: -0.02em; }
.brand-sub { font-size: 0.8125rem; color: var(--muted); margin-bottom: 14px; }
.side-section { font-size: 0.8125rem; font-weight: 600; color: var(--muted); margin: 16px 0 8px 0; }

.status-row {
    display: flex; align-items: center; justify-content: space-between; gap: 10px;
    padding: 8px 12px; margin-bottom: 6px;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid var(--border);
    border-radius: 8px;
}
.status-left { display: flex; align-items: center; gap: 8px; }
.status-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
.dot-ok { background: #34D399; }
.dot-err { background: #F97066; }
.dot-info { background: #8E9CF0; }
.status-label { font-size: 0.8125rem; color: var(--muted); }
.status-value { font-size: 0.875rem; font-weight: 700; color: var(--text); }
.status-value.err { color: #FDA29B; }
.side-version { font-size: 0.8125rem; color: var(--muted); margin-top: 10px; }

/* =====================================================
   FORM WIDGETS (always dark text on light controls)
   ===================================================== */
.stApp [data-testid="stWidgetLabel"] p,
.stApp [data-testid="stWidgetLabel"] label {
    color: var(--text-2) !important;
    font-size: 0.8125rem !important;
    font-weight: 600 !important;
}
.stApp div[data-baseweb="input"],
.stApp div[data-baseweb="textarea"],
.stApp div[data-baseweb="select"] > div {
    background: #FFFFFF !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: 8px !important;
}
.stApp div[data-baseweb="base-input"] { background: #FFFFFF !important; }
.stApp input, .stApp textarea {
    color: var(--text) !important;
    -webkit-text-fill-color: var(--text) !important;
    background: #FFFFFF !important;
    caret-color: var(--primary);
}
.stApp input::placeholder, .stApp textarea::placeholder {
    color: #667085 !important;
    -webkit-text-fill-color: #667085 !important;
    opacity: 1 !important;
}
.stApp div[data-baseweb="input"]:focus-within,
.stApp div[data-baseweb="textarea"]:focus-within,
.stApp div[data-baseweb="select"] > div:focus-within {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(76, 95, 213, 0.16) !important;
}

/* Number input steppers */
.stApp [data-testid="stNumberInput"] button {
    background: var(--surface-2) !important;
    color: var(--text-2) !important;
    border: none !important;
}
.stApp [data-testid="stNumberInput"] button:hover {
    background: #E4E7EC !important;
    color: var(--primary-dark) !important;
}
.stApp [data-testid="stNumberInput"] button svg { fill: currentColor !important; }

/* Selectbox / multiselect */
.stApp div[data-baseweb="select"] * { color: var(--text) !important; }
.stApp div[data-baseweb="select"] svg { fill: #475467 !important; }
.stApp div[data-baseweb="select"] span[data-baseweb="tag"] {
    background: var(--primary-soft) !important;
    border: 1px solid var(--primary-line) !important;
    border-radius: 6px !important;
}
.stApp div[data-baseweb="select"] span[data-baseweb="tag"],
.stApp div[data-baseweb="select"] span[data-baseweb="tag"] * { color: var(--primary-ink) !important; }
.stApp div[data-baseweb="select"] span[data-baseweb="tag"] svg { fill: var(--primary-ink) !important; }

/* Dropdown menus live in a portal outside the sidebar */
ul[role="listbox"],
div[data-baseweb="menu"],
div[data-baseweb="popover"]:has(ul[role="listbox"]) > div {
    background: #FFFFFF !important;
    color: #101828 !important;
}
div[data-baseweb="popover"]:has(ul[role="listbox"]) {
    border: 1px solid #E4E7EC;
    border-radius: 10px;
    box-shadow: 0 8px 24px rgba(16, 24, 40, 0.12);
}
li[role="option"] { background: #FFFFFF !important; color: #101828 !important; }
li[role="option"] * { color: #101828 !important; }
li[role="option"]:hover,
li[role="option"][aria-selected="true"] { background: #EEF1FD !important; }
li[role="option"]:hover *,
li[role="option"][aria-selected="true"] * { color: #2B3790 !important; }

/* Slider */
.stApp [data-testid="stSlider"] [data-testid="stSliderThumbValue"] {
    color: var(--primary-dark) !important;
    font-weight: 700;
}
.stApp [data-testid="stSliderTickBarMin"],
.stApp [data-testid="stSliderTickBarMax"] { color: var(--muted) !important; }
.stApp [data-testid="stSlider"] div[role="slider"] {
    background-color: var(--primary) !important;
    box-shadow: 0 0 0 4px rgba(76, 95, 213, 0.18) !important;
}

/* Buttons (secondary + download) */
.stApp .stButton > button,
.stApp [data-testid="stDownloadButton"] > button,
.stApp button[data-testid="stBaseButton-secondary"] {
    background: #FFFFFF !important;
    color: var(--text) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.35rem 0.9rem !important;
    box-shadow: none;
    transition: background 0.15s ease, border-color 0.15s ease, color 0.15s ease;
}
.stApp .stButton > button [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stDownloadButton"] > button [data-testid="stMarkdownContainer"] p,
.stApp button[data-testid="stBaseButton-secondary"] [data-testid="stMarkdownContainer"] p {
    color: var(--text) !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
}
.stApp .stButton > button:hover,
.stApp [data-testid="stDownloadButton"] > button:hover,
.stApp button[data-testid="stBaseButton-secondary"]:hover {
    background: var(--primary-soft) !important;
    border-color: var(--primary) !important;
}
.stApp .stButton > button:hover [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stDownloadButton"] > button:hover [data-testid="stMarkdownContainer"] p,
.stApp button[data-testid="stBaseButton-secondary"]:hover [data-testid="stMarkdownContainer"] p {
    color: var(--primary-ink) !important;
}
/* Primary: white text on accent */
.stApp .stButton > button[kind="primary"],
.stApp button[data-testid="stBaseButton-primary"] {
    background: var(--primary) !important;
    border: 1px solid var(--primary) !important;
    color: #FFFFFF !important;
}
.stApp .stButton > button[kind="primary"] [data-testid="stMarkdownContainer"] p,
.stApp button[data-testid="stBaseButton-primary"] [data-testid="stMarkdownContainer"] p {
    color: #FFFFFF !important;
}
.stApp .stButton > button[kind="primary"]:hover,
.stApp button[data-testid="stBaseButton-primary"]:hover {
    background: var(--primary-hover) !important;
    border-color: var(--primary-hover) !important;
}
.stApp .stButton > button[kind="primary"]:hover [data-testid="stMarkdownContainer"] p,
.stApp button[data-testid="stBaseButton-primary"]:hover [data-testid="stMarkdownContainer"] p {
    color: #FFFFFF !important;
}

/* =====================================================
   METRICS, ALERTS, EXPANDERS, DATAFRAME
   ===================================================== */
.stApp [data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 10px 14px;
}
.stApp [data-testid="stMetricLabel"],
.stApp [data-testid="stMetricLabel"] p,
.stApp [data-testid="stMetricLabel"] div {
    color: var(--muted) !important;
    font-size: 0.8125rem !important;
    font-weight: 600 !important;
}
.stApp [data-testid="stMetricValue"],
.stApp [data-testid="stMetricValue"] div {
    color: var(--text) !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
    font-variant-numeric: tabular-nums;
}
.stApp [data-testid="stAlert"] {
    border-radius: 10px;
    border: 1px solid rgba(16, 24, 40, 0.08);
}
.stApp [data-testid="stAlert"] :where(p, li, span, div, strong, code, a) { color: var(--text) !important; }
.stApp [data-testid="stExpander"] {
    background: #FFFFFF;
    border: 1px solid var(--border) !important;
    border-radius: 10px;
}
.stApp [data-testid="stExpander"] summary,
.stApp [data-testid="stExpander"] summary p,
.stApp [data-testid="stExpander"] summary span { color: var(--text) !important; font-weight: 600; }
.stApp [data-testid="stExpander"] summary svg { fill: var(--text-2) !important; }
.stApp [data-testid="stExpander"] [data-testid="stExpanderDetails"] { color: var(--text-2); }
[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
    --gdg-bg-cell: #FFFFFF;
    --gdg-bg-cell-medium: #F9FAFB;
    --gdg-bg-header: #F2F4F7;
    --gdg-bg-header-has-focus: #E4E7EC;
    --gdg-bg-header-hovered: #E4E7EC;
    --gdg-text-dark: #101828;
    --gdg-text-medium: #344054;
    --gdg-text-light: #667085;
    --gdg-text-header: #344054;
    --gdg-border-color: #E4E7EC;
    --gdg-header-font-style: 600 13px;
}

/* =====================================================
   CARDS (st.container(key="card_..."))
   ===================================================== */
.stApp [class*="st-key-card_"] {
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px 18px;
    box-shadow: none;
    transition: border-color 0.15s ease;
}
.stApp [class*="st-key-card_"]:hover { border-color: var(--primary-line); }
.card-title { font-size: 1rem; font-weight: 700; color: var(--text); letter-spacing: -0.02em; line-height: 1.3; }
.card-sub { font-size: 0.8125rem; color: var(--muted); margin-top: 1px; margin-bottom: 2px; }

/* =====================================================
   PAGE HEADER, BANNER, SECTIONS
   ===================================================== */
.page-header { margin-bottom: 12px; }
.page-title { font-size: 1.75rem; font-weight: 800; color: var(--text); letter-spacing: -0.02em; line-height: 1.2; }
.page-sub { font-size: 0.9375rem; color: var(--muted); margin-top: 2px; }

.hero {
    background: linear-gradient(135deg, #101828 0%, #26358F 100%);
    border-radius: 12px;
    padding: 14px 20px;
    margin-bottom: 14px;
}
.hero-title { font-size: 1.3rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.02em; line-height: 1.25; }
.hero-sub { font-size: 0.875rem; color: #D0D5DD; max-width: 680px; line-height: 1.5; margin-top: 3px; }

.section-head { margin: 2px 0 10px 0; }
.section-title { font-size: 1.2rem; font-weight: 700; color: var(--text); letter-spacing: -0.02em; }
.section-sub { font-size: 0.8125rem; color: var(--muted); margin-top: 1px; }
.divider { height: 1px; background: var(--border); margin: 18px 0; }

/* =====================================================
   KPI CARDS
   ===================================================== */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
    gap: 10px;
    margin-bottom: 14px;
}
.kpi {
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 14px;
    transition: border-color 0.15s ease;
}
.kpi:hover { border-color: var(--primary-line); }
.kpi-label { font-size: 0.8125rem; font-weight: 600; color: var(--muted); }
.kpi-value {
    font-size: 1.6rem; font-weight: 800; color: var(--text);
    letter-spacing: -0.02em; line-height: 1.2; margin-top: 3px;
    font-variant-numeric: tabular-nums;
}
.kpi-sub { font-size: 0.8125rem; color: var(--muted); margin-top: 2px; }

/* =====================================================
   BADGES, TILES, CHIPS
   ===================================================== */
.badge {
    display: inline-block; padding: 2px 10px; border-radius: 999px;
    font-size: 0.8125rem; font-weight: 600; border: 1px solid transparent;
}
.badge-success { background: #ECFDF3; color: #047857; border-color: #ABEFC6; }
.badge-warning { background: #FFFAEB; color: #B54708; border-color: #FEDF89; }
.badge-danger { background: #FEF3F2; color: #B42318; border-color: #FECDCA; }
.badge-neutral { background: #F2F4F7; color: #344054; border-color: #E4E7EC; }
.badge-primary { background: #EEF1FD; color: #2B3790; border-color: #C5CCF5; }

.tile-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }
.tile {
    background: #FFFFFF; border: 1px solid var(--border); border-radius: 10px;
    padding: 10px 12px; transition: border-color 0.15s ease;
}
.tile:hover { border-color: var(--primary-line); }
.tile-label { font-size: 0.8125rem; font-weight: 600; color: var(--muted); }
.tile-value { font-size: 1.15rem; font-weight: 700; color: var(--text); margin-top: 2px; letter-spacing: -0.01em; }

.chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 5px; }
.chip {
    background: #F2F4F7; color: #344054; border: 1px solid #E4E7EC;
    font-size: 0.8125rem; font-weight: 500; padding: 1px 8px; border-radius: 6px;
}

/* =====================================================
   LIME EXPLANATION
   ===================================================== */
.lime-card {
    background: #FFFFFF; border: 1px solid var(--border); border-radius: 12px;
    padding: 14px 18px; margin: 8px 0 4px 0;
}
.lime-title { font-size: 1rem; font-weight: 700; color: var(--text); margin-bottom: 4px; }
.lime-row {
    display: grid; grid-template-columns: minmax(150px, 1.5fr) 2fr 80px;
    gap: 12px; align-items: center; padding: 7px 0; border-bottom: 1px solid #F2F4F7;
}
.lime-feature { font-size: 0.875rem; font-weight: 600; color: var(--text); word-break: break-word; }
.lime-dir { font-size: 0.8125rem; color: var(--muted); font-weight: 400; margin-top: 1px; }
.lime-track { height: 8px; background: #F2F4F7; border-radius: 999px; overflow: hidden; }
.lime-fill { height: 100%; border-radius: 999px; }
.lime-fill.pos { background: #10B981; }
.lime-fill.neg { background: #E5484D; }
.lime-val { text-align: right; font-weight: 700; font-size: 0.9rem; }
.lime-val.pos { color: #047857; }
.lime-val.neg { color: #B42318; }
.lime-legend { font-size: 0.8125rem; color: var(--muted); margin-top: 10px; }
.lime-legend .pos { color: #047857; font-weight: 700; }
.lime-legend .neg { color: #B42318; font-weight: 700; }

/* =====================================================
   PREDICTOR: result banner, roles, notes, recommendations
   ===================================================== */
.result {
    background: linear-gradient(135deg, #101828 0%, #26358F 100%);
    border-radius: 12px; padding: 16px 20px; margin: 2px 0;
    display: flex; flex-wrap: wrap; gap: 20px; align-items: center; justify-content: space-between;
}
.result-left { min-width: 200px; }
.result-label { font-size: 0.8125rem; font-weight: 600; color: #B4BEF5; }
.result-value { font-size: 2.2rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.03em; line-height: 1.1; margin-top: 2px; }
.result-note { font-size: 0.9rem; color: #E4E7EC; margin-top: 4px; }
.result-right { flex: 1; min-width: 240px; max-width: 420px; }
.result-potential { font-size: 0.875rem; color: #D0D5DD; margin-bottom: 6px; }
.result-potential b { font-size: 1rem; font-weight: 700; }
.result-track { height: 10px; background: rgba(255, 255, 255, 0.16); border-radius: 999px; overflow: hidden; }
.result-fill { height: 100%; border-radius: 999px; }

.role-row {
    display: flex; align-items: center; gap: 14px; flex-wrap: wrap;
    background: #FFFFFF; border: 1px solid var(--border); border-radius: 10px;
    padding: 10px 14px; margin-bottom: 8px;
    transition: border-color 0.15s ease;
}
.role-row:hover { border-color: var(--primary-line); }
.role-rank {
    width: 28px; height: 28px; border-radius: 50%; flex: none;
    background: var(--primary-soft); color: var(--primary-ink); font-weight: 700; font-size: 0.875rem;
    display: flex; align-items: center; justify-content: center;
}
.role-main { flex: 1 1 220px; min-width: 0; }
.role-name { font-size: 0.9375rem; font-weight: 700; color: var(--text); }
.role-score { flex: 0 1 190px; min-width: 140px; }
.role-pct { font-size: 0.9375rem; font-weight: 700; color: var(--text); text-align: right; margin-bottom: 3px; }
.bar-track { height: 8px; background: #F2F4F7; border-radius: 999px; overflow: hidden; }
.bar-fill { height: 100%; background: linear-gradient(90deg, #6B7BEA, #4C5FD5); border-radius: 999px; }

.score-card {
    background: #F9FAFB; border: 1px solid var(--border); border-radius: 10px;
    padding: 10px 14px; margin: 8px 0;
}
.score-top { display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 6px; }
.score-label { font-size: 0.8125rem; font-weight: 600; color: var(--muted); }
.score-value { font-size: 1.35rem; font-weight: 800; letter-spacing: -0.02em; }
.note-title { font-size: 0.875rem; font-weight: 700; color: var(--text); margin: 10px 0 5px 0; }
.note {
    padding: 6px 10px; border-radius: 8px; border-left: 3px solid;
    font-size: 0.9rem; line-height: 1.45; margin-bottom: 5px;
}
.note-good { background: #ECFDF3; border-color: #10B981; color: #05603A; }
.note-warn { background: #FFFAEB; border-color: #F59E0B; color: #7A2E0E; }

.priority {
    display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
    padding: 9px 14px; border-radius: 10px; border: 1px solid; margin-bottom: 10px;
    font-size: 0.9rem; color: var(--text);
}
.priority-danger { background: #FEF3F2; border-color: #FECDCA; }
.priority-warning { background: #FFFAEB; border-color: #FEDF89; }
.priority-success { background: #ECFDF3; border-color: #ABEFC6; }
.priority-neutral { background: #F2F4F7; border-color: #E4E7EC; }

.rec-item {
    background: #FFFFFF; border: 1px solid var(--border); border-radius: 10px;
    padding: 10px 14px; margin-bottom: 8px;
    transition: border-color 0.15s ease;
}
.rec-item:hover { border-color: var(--primary-line); }
.rec-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.rec-title { font-size: 0.9375rem; font-weight: 700; color: var(--text); }
.rec-reason { font-size: 0.875rem; color: var(--muted); margin-top: 4px; line-height: 1.5; }

/* =====================================================
   EMPTY STATES & FOOTER
   ===================================================== */
.empty {
    text-align: center; padding: 18px 14px; background: #FFFFFF;
    border: 1px dashed var(--border-strong); border-radius: 12px; margin: 4px 0;
}
.empty-title { font-size: 0.9375rem; font-weight: 700; color: var(--text); }
.empty-text { font-size: 0.875rem; color: var(--muted); margin-top: 2px; }
.footer { text-align: center; color: var(--muted); font-size: 0.8125rem; padding: 2px 0 0 0; }

@media (max-width: 768px) {
    .block-container { padding: 1rem 0.85rem 1.5rem 0.85rem !important; }
    .hero { padding: 12px 14px; }
    .hero-title { font-size: 1.15rem; }
    .lime-row { grid-template-columns: 1fr; gap: 4px; }
    .result-value { font-size: 2rem; }
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# UI HELPERS (presentation only, no business logic)
# =====================================================
def esc(text):
    """Escape text before putting it inside HTML."""
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def render_html(markup, container=st):
    """Render HTML via markdown. Lines are stripped so Markdown never turns
    indented HTML into a code block, and blank lines never split an HTML block."""
    cleaned = "\n".join(line.strip() for line in markup.strip().splitlines() if line.strip())
    container.markdown(cleaned, unsafe_allow_html=True)


def page_header(title, subtitle):
    render_html(f'<div class="page-header"><div class="page-title">{esc(title)}</div>'
                f'<div class="page-sub">{esc(subtitle)}</div></div>')


def section_head(title, subtitle=""):
    sub = f'<div class="section-sub">{esc(subtitle)}</div>' if subtitle else ""
    render_html(f'<div class="section-head"><div class="section-title">{esc(title)}</div>{sub}</div>')


def card_head(title, subtitle=""):
    sub = f'<div class="card-sub">{esc(subtitle)}</div>' if subtitle else ""
    render_html(f'<div class="card-title">{esc(title)}</div>{sub}')


def divider():
    render_html('<div class="divider"></div>')


def empty_state(title, text):
    render_html(f'<div class="empty"><div class="empty-title">{esc(title)}</div>'
                f'<div class="empty-text">{esc(text)}</div></div>')


def tone_for(label):
    """Map High/Medium/Low style labels to a badge tone."""
    t = str(label).lower()
    if "high" in t:
        return "danger"
    if "med" in t or "moder" in t:
        return "warning"
    if "low" in t:
        return "success"
    return "neutral"


def score_colors(value):
    """(text colour, bar colour) for a 0-100 score."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = 0
    if v >= 70:
        return "#047857", "#10B981"
    if v >= 40:
        return "#B45309", "#F59E0B"
    return "#B42318", "#E5484D"


def risk_color_map(values):
    mapping = {}
    for v in values:
        t = str(v).lower()
        if "high" in t:
            mapping[v] = "#E5484D"
        elif "med" in t or "moder" in t:
            mapping[v] = "#F59E0B"
        elif "low" in t:
            mapping[v] = "#10B981"
        else:
            mapping[v] = "#94A3B8"
    return mapping


def kpi(label, value, sub, accent):
    return (f'<div class="kpi acc-{accent}"><div class="kpi-label">{esc(label)}</div>'
            f'<div class="kpi-value">{value}</div><div class="kpi-sub">{esc(sub)}</div></div>')


def status_row(label, value, dot="info", is_error=False):
    cls = "status-value err" if is_error else "status-value"
    return (f'<div class="status-row"><div class="status-left"><span class="status-dot dot-{dot}"></span>'
            f'<span class="status-label">{esc(label)}</span></div>'
            f'<span class="{cls}">{esc(value)}</span></div>')


def style_fig(fig, height=320, legend=True):
    """One consistent look for every Plotly chart: white surface, dark text,
    light grid, readable hover labels, shared colour palette."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family=FONT_STACK, size=12, color="#334155"),
        colorway=PALETTE,
        margin=dict(l=48, r=16, t=16, b=84 if legend else 48),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="top", y=-0.28, xanchor="center", x=0.5,
                    font=dict(color="#334155", size=11), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor="#CBD5E1",
                        font=dict(family=FONT_STACK, size=12, color="#0F172A")),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="#CBD5E1",
                     tickfont=dict(color="#475569", size=12),
                     title_font=dict(color="#334155", size=12))
    fig.update_yaxes(gridcolor="#F1F5F9", zeroline=False, showline=False,
                     tickfont=dict(color="#475569", size=12),
                     title_font=dict(color="#334155", size=12))
    return fig


def show_chart(fig):
    st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})


# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_data():
    file_path = "datadb/student_data_with_features.csv"
    if not os.path.exists(file_path):
        st.error(f"❌ Data file not found: {file_path}")
        return None
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None

# =====================================================
# LOAD MODEL - Fix for Python 3.14
# =====================================================
@st.cache_resource
def load_model():
    model_path = "models/placement_predictor_model.pkl"
    if not os.path.exists(model_path):
        st.warning(f"⚠️ Model not found: {model_path}")
        return None
    
    try:
        import joblib
        return joblib.load(model_path)
    except (ImportError, AttributeError, RuntimeError):
        try:
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            return None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# =====================================================
# LOAD MODEL METRICS
# =====================================================
@st.cache_data
def load_model_metrics():
    report_dir = 'reports/ml'
    if not os.path.exists(report_dir):
        return None
    folders = [f for f in os.listdir(report_dir) if os.path.isdir(os.path.join(report_dir, f))]
    if not folders:
        return None
    latest = sorted(folders)[-1]
    metrics_file = os.path.join(report_dir, latest, 'model_metrics.csv')
    if os.path.exists(metrics_file):
        return pd.read_csv(metrics_file)
    return None

# =====================================================
# LOAD EVERYTHING
# =====================================================
df = load_data()
model = load_model()
metrics = load_model_metrics()

if df is None:
    st.stop()

# =====================================================
# SIDEBAR
# =====================================================
render_html("""
<div class="brand">
    <div class="brand-mark">P</div>
    <div class="brand-name">Placement AI</div>
</div>
<div class="brand-sub">Placement analytics platform</div>
""", st.sidebar)

page = st.sidebar.radio(
    "Navigation",
    ["📊 Dashboard", "🤖 AI Predictor", "🔍 Student Profile", "📈 Analytics"],
    label_visibility="collapsed"
)

# System Status
render_html('<div class="side-section">System status</div>', st.sidebar)

if model is not None:
    acc_text = "Loaded"
    if metrics is not None and 'model' in metrics.columns:
        xgb_row = metrics[metrics['model'] == 'xgb']
        if not xgb_row.empty:
            acc_text = f"{xgb_row['test_accuracy'].values[0]:.2%}"
    render_html(status_row("ML model", acc_text, dot="ok"), st.sidebar)
else:
    render_html(status_row("ML model", "Not loaded", dot="err", is_error=True), st.sidebar)

render_html(status_row("Total students", f"{len(df):,}", dot="info"), st.sidebar)
render_html('<div class="side-version">Version 3.0</div>', st.sidebar)

# =====================================================
# LIME EXPLANATION HELPER
# =====================================================
def display_lime_explanation(exp):
    """Extract and display LIME explanation in a readable format"""
    try:
        # Get the explanation data
        exp_list = exp.as_list()

        # Sort by absolute weight for better readability
        sorted_exp = sorted(exp_list, key=lambda x: abs(float(x[1])), reverse=True)
        max_abs = max([abs(float(w)) for _, w in sorted_exp] + [1e-9])

        rows = []
        # Display each feature contribution
        for feature, weight in sorted_exp:
            weight_val = float(weight)
            # Determine if positive or negative influence
            influence = "Increases placement likelihood" if weight_val > 0 else "Decreases placement likelihood"
            tone = "pos" if weight_val > 0 else "neg"

            # Clean feature name (escaped: LIME names contain < and >)
            feature_clean = esc(feature.strip())
            weight_display = f"{weight_val:+.4f}".rstrip('0').rstrip('.')
            width = max(3, abs(weight_val) / max_abs * 100)

            rows.append(
                f'<div class="lime-row">'
                f'<div class="lime-feature">{feature_clean}<div class="lime-dir">{influence}</div></div>'
                f'<div class="lime-track"><div class="lime-fill {tone}" style="width:{width:.1f}%"></div></div>'
                f'<div class="lime-val {tone}">{weight_display}</div>'
                f'</div>'
            )

        render_html(
            '<div class="lime-card"><div class="lime-title">Feature contributions</div>'
            + "".join(rows) +
            '<div class="lime-legend"><span class="pos">Green</span> values increase placement likelihood, '
            '<span class="neg">red</span> values decrease it.</div></div>'
        )

    except Exception as e:
        st.error(f"Error displaying explanation: {str(e)}")
        st.info("The model explanation data is available but could not be rendered. Please try again.")

# =====================================================
# FEATURES & EXPLAINER
# =====================================================
FEATURES = ['academic_score', 'coding_score', 'aptitude_score', 
            'experience_score', 'overall_strength', 'attendance', 
            'internships', 'projects']

@st.cache_resource
def get_explainer():
    if df is None or model is None:
        return None
    sample = df.sample(min(500, len(df)), random_state=42)
    X_train = sample[FEATURES].values
    return lime.lime_tabular.LimeTabularExplainer(
        X_train, feature_names=FEATURES,
        class_names=['Not Placed', 'Placed'], mode='classification'
    )

explainer = get_explainer()

# =====================================================
# PAGE 1: DASHBOARD
# =====================================================
if page == "📊 Dashboard":
    render_html("""
    <div class="hero">
        <div class="hero-title">AI Placement Analytics System</div>
        <div class="hero-sub">Predict student placements with machine learning, track performance, find skill gaps and act on the insights.</div>
    </div>
    """)

    # Filters in sidebar
    with st.sidebar:
        render_html('<div class="side-section">Filters</div>')
        dept_filter = st.multiselect("Department", df['department'].unique(), default=df['department'].unique())
        gender_filter = st.multiselect("Gender", df['gender'].unique(), default=df['gender'].unique())
        risk_filter = st.multiselect("Risk Category", df['risk_category'].unique(), default=df['risk_category'].unique())

    filtered_df = df[df['department'].isin(dept_filter) & df['gender'].isin(gender_filter) & df['risk_category'].isin(risk_filter)]

    # KPIs
    total = len(filtered_df)
    placed = len(filtered_df[filtered_df['placement_status']=='Placed'])
    not_placed = total - placed
    rate = (placed/total*100) if total>0 else 0
    avg_cgpa = filtered_df['degree_cgpa'].mean() if total>0 else 0
    avg_coding = filtered_df['coding_score'].mean() if total>0 else 0
    avg_att = filtered_df['attendance'].mean() if total>0 else 0

    section_head("Placement overview", "Student performance and placement statistics for the selected filters")

    render_html('<div class="kpi-grid">' + "".join([
        kpi("Total students", f"{total}", "In current selection", "indigo"),
        kpi("Placement rate", f"{rate:.1f}%", f"{placed} of {total} students", "green"),
        kpi("Placed students", f"{placed}", "Offer received", "sky"),
        kpi("Not placed", f"{not_placed}", "Still searching", "rose"),
        kpi("Average CGPA", f"{avg_cgpa:.2f}", "Degree CGPA", "violet"),
        kpi("Coding score", f"{avg_coding:.1f}", "Average across students", "indigo"),
        kpi("Attendance", f"{avg_att:.1f}%", "Average across students", "sky"),
        kpi("Total projects", f"{int(filtered_df['projects'].sum())}", "Completed by students", "amber"),
    ]) + '</div>')

    # Charts
    if total == 0:
        empty_state("No students match these filters",
                    "Select at least one value in each sidebar filter to see charts and records.")
    else:
        c1, c2 = st.columns([1.2, 0.8])
        with c1:
            with st.container(key="card_trend"):
                card_head("Placement rate trend", "Illustrative sample series, not computed from your dataset")
                trend = pd.DataFrame({"Month": ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep"], "Rate": [35,42,38,55,48,62,58,79,74]})
                fig = px.line(trend, x="Month", y="Rate", markers=True,
                              labels={"Rate": "Placement rate (%)", "Month": "Month"})
                fig.update_traces(line=dict(color=PRIMARY, width=3),
                                  marker=dict(size=8, color="#FFFFFF", line=dict(width=2, color=PRIMARY)),
                                  fill="tozeroy", fillcolor="rgba(76,95,213,0.08)",
                                  hovertemplate="%{x}: %{y}%<extra></extra>")
                style_fig(fig, height=280, legend=False)
                fig.update_yaxes(title_text="Placement rate (%)", range=[0, 100])
                fig.update_xaxes(title_text="Month")
                show_chart(fig)
        with c2:
            with st.container(key="card_dept"):
                card_head("Department distribution", "Students by department")
                dept_data = filtered_df.groupby('department').size().reset_index(name='Count')
                fig = px.pie(dept_data, names='department', values='Count', hole=0.62, color_discrete_sequence=PALETTE)
                fig.update_traces(textinfo="percent", textposition="inside",
                                  marker=dict(line=dict(color="#FFFFFF", width=2)),
                                  hovertemplate="%{label}: %{value} students (%{percent})<extra></extra>")
                style_fig(fig, height=280)
                fig.update_layout(
                    margin=dict(l=10, r=10, t=10, b=50),
                    legend=dict(orientation="h", yanchor="top", y=-0.05, xanchor="center", x=0.5,
                                font=dict(color="#334155", size=12)),
                    annotations=[dict(text=f"<b>{total}</b><br>students", x=0.5, y=0.5, showarrow=False,
                                      font=dict(size=15, color="#0F172A"))]
                )
                show_chart(fig)

    # Data Explorer
    divider()
    section_head("Student data explorer", "Search, filter and export student records")
    f1, f2 = st.columns([2, 1])
    with f1:
        search = st.text_input("Search student ID", placeholder="S0001")
    with f2:
        status = st.selectbox("Placement Status", ["All", "Placed", "Not_Placed"])
    display_df = filtered_df.copy()
    if search:
        display_df = display_df[display_df['student_id'].str.contains(search, case=False)]
    if status != "All":
        display_df = display_df[display_df['placement_status'] == status]
    if display_df.empty:
        empty_state("No matching records", "Try a different student ID, status or sidebar filter.")
    else:
        st.dataframe(display_df[['student_id','gender','degree_cgpa','coding_score','placement_status','risk_category']], use_container_width=True, height=300)
        st.caption(f"Showing {len(display_df)} record(s).")
    st.download_button("Download CSV", data=display_df.to_csv(index=False), file_name="Placement_Data.csv", mime="text/csv")

# =====================================================
# PAGE 2: AI PREDICTOR - MULTI-PROJECT SUPPORT
# =====================================================
elif page == "🤖 AI Predictor":
    if model is None or explainer is None:
        st.error("❌ Model or explainer not available.")
        st.stop()

    page_header("AI placement predictor",
                "Enter your academic and skill details to get a placement estimate, matching job roles and next steps.")

    # ------------------- Initialize session state for projects -------------------
    if 'projects' not in st.session_state:
        st.session_state.projects = [{'name': '', 'description': '', 'result': None}]
    if 'project_msgs' not in st.session_state:
        st.session_state.project_msgs = {}

    # ------------------- Helper: Analyze a single project -------------------
    def analyze_project_by_index(idx):
        proj = st.session_state.projects[idx]
        name = proj['name']
        desc = proj['description']
        st.session_state.project_msgs.pop(idx, None)

        # Check if same project already analyzed in another slot
        for i, p in enumerate(st.session_state.projects):
            if i != idx and p['name'] == name and p['result'] is not None:
                st.session_state.project_msgs[idx] = f"⚠️ Same project already analyzed in Project {i+1}"
                return

        if name.strip() and desc.strip():
            try:
                result = analyze_project(name, desc)
                st.session_state.projects[idx]['result'] = result
            except Exception as e:
                st.session_state.project_msgs[idx] = f"Project analysis failed: {str(e)}"
        else:
            st.session_state.project_msgs[idx] = "Please enter both Project Name and Description to analyze."

    section_head("Project analyzer", "Describe your projects and get a quality score for each one")
    for i, proj in enumerate(st.session_state.projects):
        # Use a unique key for each project's widgets
        name_key = f"proj_{i}_name"
        desc_key = f"proj_{i}_desc"

        # Current values from session_state
        current_name = st.session_state.projects[i]['name']
        current_desc = st.session_state.projects[i]['description']

        # Project card
        with st.container(key=f"card_project_{i}"):
            card_head(f"Project {i+1}")

            # Name input
            new_name = st.text_input(
                "Project Name",
                value=current_name,
                key=name_key,
                placeholder="e.g., AI Placement Analytics System"
            )
            # Description text area
            new_desc = st.text_area(
                "Project Description",
                value=current_desc,
                key=desc_key,
                placeholder="Describe your project in detail...",
                height=80
            )

            # Update session_state with new values (so they persist)
            st.session_state.projects[i]['name'] = new_name
            st.session_state.projects[i]['description'] = new_desc

            # Buttons row
            col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 4])
            with col_btn1:
                if st.button("Analyze", key=f"analyze_{i}"):
                    with st.spinner("Analyzing project..."):
                        analyze_project_by_index(i)
                    st.rerun()
            with col_btn2:
                if i > 0:  # Can't remove the first project
                    if st.button("Remove", key=f"remove_{i}"):
                        st.session_state.projects.pop(i)
                        st.session_state.project_msgs = {}
                        st.rerun()

            # Display analysis result if exists
            if st.session_state.project_msgs.get(i):
                st.warning(st.session_state.project_msgs[i])

            if st.session_state.projects[i]['result'] is not None:
                result = st.session_state.projects[i]['result']
                quality = result.get('quality_score', 0)
                strengths = result.get('strengths', [])
                improvements = result.get('improvements', [])

                text_color, bar_color = score_colors(quality)
                try:
                    bar_width = max(0, min(100, float(quality)))
                except (TypeError, ValueError):
                    bar_width = 0

                parts = [
                    f'<div class="score-card"><div class="score-top">'
                    f'<span class="score-label">Quality score</span>'
                    f'<span class="score-value" style="color:{text_color}">{esc(quality)}/100</span></div>'
                    f'<div class="bar-track"><div class="bar-fill" style="width:{bar_width}%; background:{bar_color}"></div></div></div>'
                ]
                if strengths:
                    parts.append('<div class="note-title">Strengths</div>')
                    parts.extend(f'<div class="note note-good">{esc(s)}</div>' for s in strengths)
                if improvements:
                    parts.append('<div class="note-title">Improvements</div>')
                    parts.extend(f'<div class="note note-warn">{esc(imp)}</div>' for imp in improvements)
                render_html("".join(parts))

    # ------------------- Add Project button -------------------
    if st.button("Add project"):
        st.session_state.projects.append({'name': '', 'description': '', 'result': None})
        st.rerun()

    divider()

    # ------------------- Academic & Skills Inputs -------------------
    section_head("Academic profile", "Scores and attendance used by the prediction model")
    with st.container(key="card_academic"):
        col1, col2 = st.columns(2)
        with col1:
            ssc = st.slider("SSC %", 0.0, 100.0, 75.0)
            hsc = st.slider("HSC %", 0.0, 100.0, 70.0)
            cgpa = st.slider("CGPA", 4.0, 10.0, 7.5)
            att = st.slider("Attendance %", 0.0, 100.0, 85.0)
        with col2:
            coding = st.slider("Coding Score", 0, 100, 70)
            aptitude = st.slider("Aptitude Score", 0, 100, 70)
            internships = st.number_input("Internships", 0, 10, 1)
            projects_count = st.number_input("Projects (total)", 0, 10, 2)  # kept for compatibility

    # Skills section
    st.write("")
    section_head("Your skills", "Used to match job roles and tailor recommendations")
    COMMON_SKILLS = [
        "Python", "SQL", "Machine Learning", "Deep Learning", "NLP",
        "Computer Vision", "Data Analysis", "Data Visualization", "Pandas",
        "NumPy", "Scikit-learn", "TensorFlow", "PyTorch", "Keras",
        "Git", "GitHub", "Docker", "Kubernetes", "AWS", "Azure",
        "GCP", "Linux", "Java", "C++", "JavaScript", "React", "Node.js",
        "Flask", "Django", "FastAPI", "Tableau", "Power BI", "Excel",
        "Statistics", "Probability", "Linear Algebra", "Calculus",
        "R", "Spark", "Hadoop", "Airflow", "MLflow", "Streamlit"
    ]
    with st.container(key="card_skills"):
        selected_skills = st.multiselect(
            "Select your skills (from the list)",
            options=COMMON_SKILLS,
            default=["Python", "SQL"]
        )
        extra_skills = st.text_input(
            "Add additional skills (comma separated)",
            placeholder="e.g., React, Tableau, Spark"
        )

    st.write("")

    # ------------------- Main Predict Button -------------------
    current_inputs = (ssc, hsc, cgpa, att, coding, aptitude, internships, projects_count,
                      tuple(selected_skills), extra_skills)
    if st.button("Predict", type="primary"):
        st.session_state.pred_inputs = current_inputs

    if st.session_state.get("pred_inputs") == current_inputs:
        # Combine skills
        skills_list = selected_skills.copy()
        if extra_skills.strip():
            extra_parsed = [s.strip() for s in extra_skills.split(',') if s.strip()]
            skills_list.extend(extra_parsed)
        skills_list = list(dict.fromkeys(skills_list))

        divider()

        # ----- 1. ML Prediction -----
        with st.spinner("Running prediction..."):
            academic = (ssc*0.2)+(hsc*0.3)+((cgpa/10)*100*0.5)
            experience = (internships*3)+(projects_count*2)
            overall = (academic*0.4)+(coding*0.3)+(aptitude*0.2)+(experience*0.1)
            features = np.array([academic, coding, aptitude, experience, overall, att, internships, projects_count]).reshape(1,-1)
            pred = model.predict(features)[0]
            proba = model.predict_proba(features)[0][1] if hasattr(model, 'predict_proba') else 0.5
            prob_percent = proba * 100

        # Display Prediction
        section_head("Placement prediction")
        if prob_percent >= 70:
            potential = "High"
            bar_color = "#34D399"
        elif prob_percent >= 40:
            potential = "Medium"
            bar_color = "#FBBF24"
        else:
            potential = "Low"
            bar_color = "#FB7185"
        verdict = "High chance of placement." if pred == 1 else "Needs improvement before applying."
        fill_width = max(0, min(100, prob_percent))
        render_html(f"""
        <div class="result">
            <div class="result-left">
                <div class="result-label">Placement probability</div>
                <div class="result-value">{prob_percent:.1f}%</div>
                <div class="result-note">{verdict}</div>
            </div>
            <div class="result-right">
                <div class="result-potential">Placement potential: <b style="color:{bar_color}">{potential}</b></div>
                <div class="result-track"><div class="result-fill" style="width:{fill_width:.1f}%; background:{bar_color}"></div></div>
            </div>
        </div>
        """)

        divider()

        # ----- 2. Job Role Matcher -----
        roles_result = []
        if skills_list:
            try:
                with st.spinner("Matching job roles..."):
                    roles_result = suggest_roles(skills_list, top_n=5)
            except Exception as e:
                st.warning(f"Job role matching failed: {str(e)}")

        section_head("Best-fit job roles", "Ranked by how well your skills match each role")
        if roles_result:
            for idx, role in enumerate(roles_result, 1):
                chips = ""
                if role.get('matched_skills'):
                    chips = ('<div class="chips">' + "".join(
                        f'<span class="chip">{esc(s)}</span>' for s in role['matched_skills'][:3]) + '</div>')
                try:
                    width = max(0, min(100, float(role['match_score'])))
                except (TypeError, ValueError):
                    width = 0
                render_html(f"""
                <div class="role-row">
                    <div class="role-rank">{idx}</div>
                    <div class="role-main"><div class="role-name">{esc(role['role'])}</div>{chips}</div>
                    <div class="role-score">
                        <div class="role-pct">{role['match_score']:.1f}%</div>
                        <div class="bar-track"><div class="bar-fill" style="width:{width:.1f}%"></div></div>
                    </div>
                </div>
                """)
        else:
            if not skills_list:
                empty_state("No skills provided", "Add skills above to get job role suggestions.")
            else:
                empty_state("No matching job roles found", "Try adding more skills or different keywords.")

        divider()

        # ----- 3. Aggregate Project Score -----
        # Compute average of analyzed projects
        project_scores = []
        for proj in st.session_state.projects:
            if proj['result'] is not None:
                project_scores.append(proj['result'].get('quality_score', 0))
        if project_scores:
            avg_project_score = sum(project_scores) / len(project_scores)
        else:
            avg_project_score = 0

        # Display average project quality (if any)
        section_head("Aggregate project analysis")
        if project_scores:
            st.metric("Average Project Quality Score", f"{avg_project_score:.1f}/100")
            # Show number of analyzed projects
            st.caption(f"Based on {len(project_scores)} analyzed project(s).")
        else:
            empty_state("No projects analyzed yet", "Use the Analyze button on a project to get its quality score.")
        divider()

        # ----- 4. Recommendation Engine -----
        rec_result = None
        try:
            rec_skills = skills_list if skills_list else []
            rec_job_roles = roles_result if roles_result else []

            with st.spinner("Generating recommendations..."):
                rec_result = generate_recommendations(
                    placement_probability=prob_percent,
                    skills=rec_skills,
                    job_role_matches=rec_job_roles,
                    project_score=avg_project_score,
                    internships=internships,
                    coding_score=coding,
                    aptitude_score=aptitude,
                    cgpa=cgpa,
                    attendance=att
                )
        except Exception as e:
            st.warning(f"Recommendation generation failed: {str(e)}")

        section_head("Personalized recommendations", "Suggested next steps, most important first")
        if rec_result:
            priority = rec_result.get("priority", "Low")
            if priority == "High":
                p_msg = "Focus on these areas first."
            elif priority == "Medium":
                p_msg = "Good, but there is room for improvement."
            else:
                p_msg = "You're on the right track!"
            p_tone = tone_for(priority)
            render_html(f'<div class="priority priority-{p_tone}"><span class="badge badge-{p_tone}">'
                        f'Priority: {esc(priority)}</span><span>{esc(p_msg)}</span></div>')

            if rec_result.get("strengths"):
                st.markdown("**Strengths**")
                render_html("".join(f'<div class="note note-good">{esc(s)}</div>' for s in rec_result["strengths"]))

            recs = rec_result.get("recommendations", [])
            if recs:
                st.markdown("**Recommendations**")
                for idx, rec in enumerate(recs, 1):
                    r_tone = tone_for(rec['priority'])
                    render_html(f"""
                    <div class="rec-item">
                        <div class="rec-head">
                            <div class="rec-title">{idx}. {esc(rec['title'])}</div>
                            <span class="badge badge-{r_tone}">{esc(rec['priority'])}</span>
                        </div>
                        <div class="rec-reason">{esc(rec['reason'])}</div>
                    </div>
                    """)
            else:
                empty_state("No specific recommendations", "Nothing to flag at this time.")
        else:
            empty_state("Recommendations unavailable", "They could not be generated. Please check your inputs.")

        # ----- LIME Explanation (optional) -----
        divider()
        if st.button("Explain this prediction with LIME"):
            with st.spinner("Generating explanation..."):
                exp = explainer.explain_instance(features[0], model.predict_proba, num_features=len(FEATURES))
                display_lime_explanation(exp)

    elif "pred_inputs" in st.session_state:
        st.caption("Inputs changed since the last prediction. Click Predict to refresh the results.")

# =====================================================
# PAGE 3: STUDENT PROFILE
# =====================================================
elif page == "🔍 Student Profile":
    if df is None or model is None or explainer is None:
        st.error("❌ Data, model, or explainer not available.")
        st.stop()
    page_header("Student profile", "Individual performance, comparison with the class and model explanation")
    selected_id = st.selectbox("Select Student", df['student_id'].unique())
    row = df[df['student_id']==selected_id].iloc[0]

    status_tone = "success" if str(row["placement_status"]) == "Placed" else "warning"
    risk_tone = tone_for(row["risk_category"])

    def tile(label, value_html):
        return f'<div class="tile"><div class="tile-label">{esc(label)}</div><div class="tile-value">{value_html}</div></div>'

    col1, col2 = st.columns([1, 1.15])
    with col1:
        render_html('<div class="tile-grid">' + "".join([
            tile("CGPA", f'{row["degree_cgpa"]:.2f}'),
            tile("Coding score", f'{row["coding_score"]:.1f}'),
            tile("Aptitude score", f'{row["aptitude_score"]:.1f}'),
            tile("Attendance", f'{row["attendance"]:.1f}%'),
            tile("Internships", f'{int(row["internships"])}'),
            tile("Projects", f'{int(row["projects"])}'),
            tile("Risk category", f'<span class="badge badge-{risk_tone}">{esc(row["risk_category"])}</span>'),
            tile("Placement status", f'<span class="badge badge-{status_tone}">{esc(row["placement_status"])}</span>'),
        ]) + '</div>')
    with col2:
        with st.container(key="card_compare"):
            card_head("Student vs class average", "Selected student compared with all students")
            class_avg = df[['degree_cgpa','coding_score','aptitude_score','attendance']].mean()
            student_vals = [row['degree_cgpa'], row['coding_score'], row['aptitude_score'], row['attendance']]
            cats = ['CGPA','Coding','Aptitude','Attendance']
            fig = go.Figure()
            fig.add_trace(go.Bar(x=cats, y=student_vals, name='Student', marker_color=PRIMARY,
                                 text=[f"{v:.1f}" for v in student_vals], textposition="outside",
                                 textfont=dict(color="#334155", size=12), cliponaxis=False))
            fig.add_trace(go.Bar(x=cats, y=list(class_avg), name='Class average', marker_color="#CBD5E1",
                                 text=[f"{v:.1f}" for v in class_avg], textposition="outside",
                                 textfont=dict(color="#334155", size=12), cliponaxis=False))
            style_fig(fig, height=300)
            fig.update_layout(barmode='group', bargap=0.3, margin=dict(l=48, r=16, t=30, b=84))
            fig.update_yaxes(title_text="Score")
            fig.update_xaxes(title_text="Metric")
            show_chart(fig)

    divider()
    section_head("What influenced this prediction?", "LIME shows which features pushed the model toward or away from Placed")
    if st.button("Generate LIME Explanation"):
        with st.spinner("Computing..."):
            exp = explainer.explain_instance(row[FEATURES].values, model.predict_proba, num_features=len(FEATURES))
            display_lime_explanation(exp)

# =====================================================
# PAGE 4: ANALYTICS (with ML Validation Dashboard)
# =====================================================
elif page == "📈 Analytics":
    page_header("Analytics", "Model validation, feature importance and dataset statistics")

    # Non-interactive backend: must be selected before pyplot is imported
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, roc_curve, confusion_matrix
    )

    section_head("Model performance", "Evaluated on a 30% hold-out split of the dataset")

    target_col = "placement_status"
    missing_cols = [c for c in FEATURES if c not in df.columns]
    eval_ok = False

    if model is None:
        empty_state("Model not available", "Load a trained model to see validation metrics and feature importance.")
    elif target_col not in df.columns:
        st.warning("Target column 'placement_status' not found. Cannot compute validation metrics.")
    elif missing_cols:
        st.warning("Missing feature columns in the dataset: " + ", ".join(missing_cols))
    else:
        # ---------- Evaluate (every step guarded) ----------
        try:
            with st.spinner("Evaluating model..."):
                y_all = df[target_col].map({"Placed": 1, "Not_Placed": 0})
                valid = y_all.notna()
                X_all = df.loc[valid, FEATURES].apply(pd.to_numeric, errors="coerce")
                y_all = y_all[valid].astype(int)
                complete = X_all.notna().all(axis=1)
                X_all, y_all = X_all[complete], y_all[complete]
                if y_all.nunique() < 2 or y_all.value_counts().min() < 2:
                    raise ValueError("Need at least two students in both Placed and Not_Placed to evaluate the model.")

                X_train, X_test, y_train, y_test = train_test_split(
                    X_all, y_all, test_size=0.3, random_state=42, stratify=y_all
                )

                y_pred = np.asarray(model.predict(X_test))
                if y_pred.dtype.kind in "OUS":  # model returns text labels
                    y_pred = pd.Series(y_pred).map({"Placed": 1, "Not_Placed": 0}).to_numpy()

                y_proba = None
                if hasattr(model, "predict_proba"):
                    try:
                        y_proba = model.predict_proba(X_test)[:, 1]
                    except Exception:
                        y_proba = None

                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, zero_division=0)
                rec = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                auc = None
                if y_proba is not None:
                    try:
                        auc = roc_auc_score(y_test, y_proba)
                    except Exception:
                        auc = None
            eval_ok = True
        except Exception as e:
            st.warning(f"Could not evaluate the model on this dataset: {e}")

        if eval_ok:
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Accuracy", f"{acc:.2%}")
            col2.metric("Precision", f"{prec:.2%}")
            col3.metric("Recall", f"{rec:.2%}")
            col4.metric("F1 Score", f"{f1:.2f}")
            col5.metric("ROC-AUC", f"{auc:.3f}" if auc is not None else "N/A")

            st.write("")
            cm_col, roc_col = st.columns(2)
            with cm_col:
                with st.container(key="card_confusion"):
                    card_head("Confusion matrix", "Actual vs predicted classes on the hold-out set")
                    try:
                        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
                        fig_cm = px.imshow(
                            cm,
                            text_auto=True,
                            labels=dict(x="Predicted", y="Actual"),
                            x=["Not Placed", "Placed"],
                            y=["Not Placed", "Placed"],
                            color_continuous_scale=[[0, "#EEF2FF"], [1, "#4C5FD5"]],
                        )
                        style_fig(fig_cm, height=320, legend=False)
                        fig_cm.update_layout(coloraxis_showscale=False, margin=dict(l=70, r=16, t=16, b=60))
                        fig_cm.update_traces(textfont=dict(size=18))
                        fig_cm.update_xaxes(title_text="Predicted")
                        fig_cm.update_yaxes(title_text="Actual")
                        show_chart(fig_cm)
                    except Exception as e:
                        empty_state("Confusion matrix unavailable", str(e))
            with roc_col:
                with st.container(key="card_roc"):
                    if y_proba is not None and auc is not None:
                        card_head("ROC curve", f"Area under curve: {auc:.3f}")
                        try:
                            fpr, tpr, _ = roc_curve(y_test, y_proba)
                            fig_roc = px.line(
                                x=fpr, y=tpr,
                                labels={"x": "False positive rate", "y": "True positive rate"}
                            )
                            fig_roc.update_traces(line=dict(color=PRIMARY, width=3), fill="tozeroy",
                                                  fillcolor="rgba(76,95,213,0.08)",
                                                  hovertemplate="FPR %{x:.2f}, TPR %{y:.2f}<extra></extra>")
                            fig_roc.add_shape(
                                type="line", line=dict(dash="dash", color="#94A3B8"),
                                x0=0, y0=0, x1=1, y1=1
                            )
                            style_fig(fig_roc, height=320, legend=False)
                            fig_roc.update_xaxes(title_text="False positive rate", range=[0, 1])
                            fig_roc.update_yaxes(title_text="True positive rate", range=[0, 1.02])
                            show_chart(fig_roc)
                        except Exception as e:
                            empty_state("ROC curve unavailable", str(e))
                    else:
                        card_head("ROC curve")
                        empty_state("Not available", "This model does not expose usable predicted probabilities.")

            # ---------- Feature importance: SHAP first, built-in importance as fallback ----------
            st.write("")
            with st.container(key="card_shap"):
                card_head("Feature importance", "How much each feature drives the model's predictions")
                shap_error = None
                try:
                    import shap  # may fail if numba/numpy versions clash
                    with st.spinner("Computing SHAP values (may take a few seconds)..."):
                        X_sample = X_train.sample(min(100, len(X_train)), random_state=42)
                        shap_explainer = shap.TreeExplainer(model)
                        shap_values = shap_explainer.shap_values(X_sample)
                        if isinstance(shap_values, list):
                            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                        else:
                            shap_values = np.asarray(shap_values)
                            if shap_values.ndim == 3:
                                shap_values = shap_values[:, :, 1] if shap_values.shape[2] > 1 else shap_values[:, :, 0]

                        plt.rcParams.update({
                            "text.color": "#334155", "axes.labelcolor": "#334155",
                            "xtick.color": "#475569", "ytick.color": "#475569",
                            "axes.edgecolor": "#CBD5E1", "font.family": "sans-serif",
                        })
                        shap_fig, shap_ax = plt.subplots(figsize=(9, 4.2))
                        shap_fig.patch.set_facecolor("#FFFFFF")
                        shap_ax.set_facecolor("#FFFFFF")
                        shap.summary_plot(shap_values, X_sample, feature_names=FEATURES, show=False, plot_size=None)
                        plt.tight_layout()
                        st.pyplot(shap_fig)
                        plt.close(shap_fig)
                    st.caption("SHAP values show how each feature contributes to the prediction (red = high feature value, blue = low).")
                except Exception as e:
                    shap_error = e
                    try:
                        plt.close("all")
                    except Exception:
                        pass

                if shap_error is not None:
                    st.info("SHAP is not available in this environment, so the model's built-in feature importance is shown instead.")
                    importances = getattr(model, "feature_importances_", None)
                    if importances is None and hasattr(model, "coef_"):
                        try:
                            importances = np.abs(np.asarray(model.coef_)).ravel()
                        except Exception:
                            importances = None
                    if importances is not None and len(importances) == len(FEATURES):
                        imp_df = pd.DataFrame({
                            "Feature": [f.replace("_", " ").title() for f in FEATURES],
                            "Importance": np.asarray(importances, dtype=float),
                        }).sort_values("Importance")
                        fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation="h")
                        fig_imp.update_traces(marker_color=PRIMARY,
                                              hovertemplate="%{y}: %{x:.3f}<extra></extra>")
                        style_fig(fig_imp, height=320, legend=False)
                        fig_imp.update_xaxes(title_text="Importance")
                        fig_imp.update_yaxes(title_text="")
                        show_chart(fig_imp)
                    else:
                        st.caption("This model type does not expose built-in feature importances.")
                    with st.expander("Technical details"):
                        st.write("To restore SHAP, repair the numba install in your virtual environment, "
                                 "for example: `pip install --upgrade numba llvmlite shap`.")
                        st.code(f"{type(shap_error).__name__}: {shap_error}")

    divider()
    section_head("Dataset overview", "Feature relationships and population breakdown")

    # ---------- Correlation heatmap ----------
    with st.container(key="card_corr"):
        card_head("Feature correlation", "Pearson correlation between numeric features (1 = move together, -1 = move opposite)")
        try:
            corr_cols = [c for c in dict.fromkeys(FEATURES + ['degree_cgpa', 'coding_score', 'aptitude_score'])
                         if c in df.columns]
            corr_src = df[corr_cols].apply(pd.to_numeric, errors="coerce")
            corr_src = corr_src.loc[:, corr_src.notna().any() & (corr_src.nunique() > 1)]
            if corr_src.shape[1] < 2:
                empty_state("Not enough numeric data", "At least two varying numeric columns are needed for a correlation matrix.")
            else:
                corr_df = corr_src.corr()
                labels = [c.replace("_", " ").title() for c in corr_df.columns]
                fig_corr = go.Figure(data=go.Heatmap(
                    z=corr_df.values, x=labels, y=labels,
                    colorscale=[[0, "#E5484D"], [0.5, "#FFFFFF"], [1, "#4C5FD5"]],
                    zmin=-1, zmax=1, texttemplate="%{z:.2f}", textfont=dict(size=11),
                    xgap=2, ygap=2,
                    colorbar=dict(title=dict(text="Correlation", font=dict(color="#334155", size=12)),
                                  tickfont=dict(color="#475569"), thickness=12, outlinewidth=0),
                    hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>"
                ))
                style_fig(fig_corr, height=400, legend=False)
                fig_corr.update_layout(margin=dict(l=130, r=16, t=16, b=110))
                fig_corr.update_xaxes(tickangle=-40, showgrid=False)
                fig_corr.update_yaxes(autorange="reversed", showgrid=False)
                show_chart(fig_corr)
        except Exception as e:
            empty_state("Correlation heatmap unavailable", str(e))

    # ---------- Distribution donuts ----------
    st.write("")
    col1, col2 = st.columns(2)
    with col1:
        with st.container(key="card_pie_dept"):
            card_head("Department distribution", "Share of students by department")
            if 'department' in df.columns and df['department'].notna().any():
                fig_dept = px.pie(df, names='department', hole=0.55, color_discrete_sequence=PALETTE)
                fig_dept.update_traces(textinfo="percent", marker=dict(line=dict(color="#FFFFFF", width=2)),
                                       hovertemplate="%{label}: %{value} students (%{percent})<extra></extra>")
                style_fig(fig_dept, height=290)
                fig_dept.update_layout(margin=dict(l=10, r=10, t=10, b=50),
                                       legend=dict(orientation="h", yanchor="top", y=-0.05, xanchor="center", x=0.5,
                                                   font=dict(color="#334155", size=11)))
                show_chart(fig_dept)
            else:
                empty_state("No department data", "The 'department' column is missing or empty.")
    with col2:
        with st.container(key="card_pie_risk"):
            card_head("Risk category", "Share of students by placement risk")
            if 'risk_category' in df.columns and df['risk_category'].notna().any():
                fig_risk = px.pie(df, names='risk_category', hole=0.55,
                                  color='risk_category',
                                  color_discrete_map=risk_color_map(df['risk_category'].dropna().unique()))
                fig_risk.update_traces(textinfo="percent", marker=dict(line=dict(color="#FFFFFF", width=2)),
                                       hovertemplate="%{label}: %{value} students (%{percent})<extra></extra>")
                style_fig(fig_risk, height=290)
                fig_risk.update_layout(margin=dict(l=10, r=10, t=10, b=50),
                                       legend=dict(orientation="h", yanchor="top", y=-0.05, xanchor="center", x=0.5,
                                                   font=dict(color="#334155", size=11)))
                show_chart(fig_risk)
            else:
                empty_state("No risk data", "The 'risk_category' column is missing or empty.")

# =====================================================
# FOOTER
# =====================================================
divider()
render_html('<div class="footer">© 2026 Placement AI Analytics Platform · Built with Streamlit</div>')
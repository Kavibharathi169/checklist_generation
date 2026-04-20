import streamlit as st
import time
import json
import re
import requests
import os
import uuid
import pandas as pd

API_URL = os.getenv("API_URL", "http://localhost:8000")
HTTP_TIMEOUT_SEC = float(os.getenv("API_TIMEOUT_SEC", "60"))
STREAM_CONNECT_TIMEOUT_SEC = float(os.getenv("API_STREAM_CONNECT_TIMEOUT_SEC", "20"))
STREAM_READ_TIMEOUT_SEC = float(os.getenv("API_STREAM_READ_TIMEOUT_SEC", "300"))

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GovCheck AI",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "ui_theme" not in st.session_state:
    st.session_state.ui_theme = "dark"

# ── Global CSS ────────────────────────────────────────────────────────────────
if st.session_state.ui_theme == "light":
    theme_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root & base (LIGHT) ── */
:root {
    --teal:        #00a884;
    --teal-dim:    rgba(0,168,132,0.18);
    --teal-glow:   rgba(0,168,132,0.06);
    --blue:        #2563eb;
    --blue-dim:    rgba(37,99,235,0.18);
    --amber:       #d97706;
    --amber-dim:   rgba(217,119,6,0.18);
    --purple:      #9333ea;
    --purple-dim:  rgba(147,51,234,0.18);
    --coral:       #e11d48;
    --coral-dim:   rgba(225,29,72,0.18);
    --glass-bg:    rgba(255,255,255,0.7);
    --glass-bdr:   rgba(0,0,0,0.10);
    --glass-bdr2:  rgba(0,0,0,0.16);
    --text-pri:    rgba(15,23,42,0.95);
    --text-sec:    rgba(15,23,42,0.65);
    --text-ter:    rgba(15,23,42,0.40);
    --sidebar-bg:  rgba(255,255,255,0.70);
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background: #eef2f6 !important;
    font-family: 'Outfit', sans-serif !important;
    color: var(--text-pri) !important;
}

/* Mesh gradient background (LIGHT) */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 60% at 10% 0%,   rgba(37,99,235,0.05) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 90% 10%,  rgba(0,168,132,0.05) 0%, transparent 60%),
        radial-gradient(ellipse 50% 60% at 50% 100%, rgba(217,119,6,0.05) 0%, transparent 60%);
    pointer-events: none;
    z-index: 0;
}
"""
else:
    theme_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root & base ── */
:root {
    --teal:        #0ff2c8;
    --teal-dim:    rgba(15,242,200,0.18);
    --teal-glow:   rgba(15,242,200,0.06);
    --blue:        #5b8ef0;
    --blue-dim:    rgba(91,142,240,0.18);
    --amber:       #f5c542;
    --amber-dim:   rgba(245,197,66,0.18);
    --purple:      #c084fc;
    --purple-dim:  rgba(192,132,252,0.18);
    --coral:       #fb7185;
    --coral-dim:   rgba(251,113,133,0.18);
    --glass-bg:    rgba(255,255,255,0.04);
    --glass-bdr:   rgba(255,255,255,0.10);
    --glass-bdr2:  rgba(255,255,255,0.16);
    --text-pri:    rgba(255,255,255,0.95);
    --text-sec:    rgba(255,255,255,0.55);
    --text-ter:    rgba(255,255,255,0.30);
    --sidebar-bg:  rgba(6,12,24,0.7);
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background: #060c18 !important;
    font-family: 'Outfit', sans-serif !important;
    color: var(--text-pri) !important;
}

/* Mesh gradient background (DARK) */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 60% at 10% 0%,   rgba(15,242,200,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 90% 10%,  rgba(91,142,240,0.14) 0%, transparent 60%),
        radial-gradient(ellipse 50% 60% at 50% 100%, rgba(192,132,252,0.10) 0%, transparent 60%);
    pointer-events: none;
    z-index: 0;
}
"""

st.markdown(theme_css + """
/* Kill default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebarCollapseButton"] { display: none; }
section[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }

/* ── Sidebar glass ── */
section[data-testid="stSidebar"] {
    background: var(--sidebar-bg) !important;
    backdrop-filter: blur(24px) !important;
    border-right: 1px solid var(--glass-bdr) !important;
}
section[data-testid="stSidebar"] * { color: var(--text-pri) !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--glass-bg) !important;
    border: 1.5px dashed var(--teal-dim) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(12px) !important;
    transition: border-color 0.2s, background 0.2s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--teal) !important;
    background: var(--teal-glow) !important;
}
[data-testid="stFileUploader"] label { color: var(--text-sec) !important; }
[data-testid="stFileUploader"] button {
    background: var(--teal-dim) !important;
    border: 1px solid rgba(15,242,200,0.3) !important;
    color: var(--teal) !important;
    border-radius: 8px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 500 !important;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background: var(--glass-bg) !important;
    border: 1px solid var(--glass-bdr) !important;
    border-radius: 10px !important;
    backdrop-filter: blur(12px) !important;
    color: var(--text-pri) !important;
}
[data-testid="stSelectbox"] svg { fill: var(--text-sec) !important; }

/* ── Text input ── */
[data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea {
    background: var(--glass-bg) !important;
    border: 1px solid var(--glass-bdr) !important;
    border-radius: 10px !important;
    backdrop-filter: blur(12px) !important;
    color: var(--text-pri) !important;
    font-family: 'Outfit', sans-serif !important;
}
[data-testid="stTextInput"] input:focus, [data-testid="stTextArea"] textarea:focus {
    border-color: rgba(15,242,200,0.5) !important;
    box-shadow: 0 0 0 3px rgba(15,242,200,0.08) !important;
}

/* ── Progress bar ── */
[data-testid="stProgress"] > div > div > div {
    background: linear-gradient(90deg, var(--teal), var(--blue)) !important;
    border-radius: 99px !important;
}
[data-testid="stProgress"] > div > div {
    background: rgba(255,255,255,0.08) !important;
    border-radius: 99px !important;
}

/* ── Buttons ── */
.stButton > button {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 500 !important;
    border-radius: 10px !important;
    transition: all 0.2s !important;
    border: 1px solid var(--glass-bdr2) !important;
    background: var(--glass-bg) !important;
    color: var(--text-pri) !important;
    backdrop-filter: blur(12px) !important;
}
.stButton > button:hover {
    background: rgba(255,255,255,0.10) !important;
    border-color: var(--teal) !important;
    color: var(--teal) !important;
    box-shadow: 0 0 16px rgba(15,242,200,0.15) !important;
}

/* Primary action button */
button[kind="primary"] {
    background: linear-gradient(135deg, rgba(15,242,200,0.25), rgba(91,142,240,0.20)) !important;
    border-color: rgba(15,242,200,0.5) !important;
    color: var(--teal) !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
}
button[kind="primary"]:hover {
    background: linear-gradient(135deg, rgba(15,242,200,0.35), rgba(91,142,240,0.30)) !important;
    box-shadow: 0 0 24px rgba(15,242,200,0.25) !important;
}

/* Export buttons */
[data-testid="stDownloadButton"] button {
    font-size: 13px !important;
    padding: 6px 14px !important;
    height: auto !important;
}

/* ── Checkbox ── */
[data-testid="stCheckbox"] label span { color: var(--text-sec) !important; font-size: 13px !important; }
[data-testid="stCheckbox"] input:checked + div {
    background: var(--teal) !important;
    border-color: var(--teal) !important;
}

/* ── Divider ── */
hr { border-color: var(--glass-bdr) !important; margin: 12px 0 !important; }

/* ── Metric ── */
[data-testid="stMetric"] {
    background: var(--glass-bg) !important;
    border: 1px solid var(--glass-bdr) !important;
    border-radius: 14px !important;
    backdrop-filter: blur(16px) !important;
    padding: 16px 20px !important;
}
[data-testid="stMetricLabel"] { color: var(--text-sec) !important; font-size: 11px !important; letter-spacing: 0.5px !important; }
[data-testid="stMetricValue"] { color: var(--text-pri) !important; font-family: 'Outfit', sans-serif !important; font-weight: 600 !important; }
[data-testid="stMetricDelta"] { font-size: 11px !important; }

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {
    background: var(--glass-bg) !important;
    border: 1px solid var(--glass-bdr) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    backdrop-filter: blur(12px) !important;
    gap: 2px !important;
}
[data-testid="stTabs"] button[role="tab"] {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    color: var(--text-sec) !important;
    border: none !important;
    background: transparent !important;
    transition: all 0.2s !important;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    background: rgba(255,255,255,0.10) !important;
    color: var(--text-pri) !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: var(--glass-bg) !important;
    border: 1px solid var(--glass-bdr) !important;
    border-radius: 12px !important;
    backdrop-filter: blur(12px) !important;
    overflow: hidden !important;
}
[data-testid="stExpander"] summary {
    color: var(--text-pri) !important;
    font-weight: 500 !important;
    font-size: 13px !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    background: var(--glass-bg) !important;
    border: 1px solid var(--glass-bdr) !important;
    border-radius: 12px !important;
    backdrop-filter: blur(12px) !important;
    overflow: hidden !important;
}

/* ── Custom glass card ── */
.glass-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 16px;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    padding: 20px 22px;
    margin-bottom: 12px;
    position: relative;
    overflow: hidden;
}
.glass-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.18), transparent);
}
.glass-card-teal  { border-color: rgba(15,242,200,0.22);  background: rgba(15,242,200,0.04); }
.glass-card-blue  { border-color: rgba(91,142,240,0.22);  background: rgba(91,142,240,0.04); }
.glass-card-amber { border-color: rgba(245,197,66,0.22);  background: rgba(245,197,66,0.04); }
.glass-card-purple{ border-color: rgba(192,132,252,0.22); background: rgba(192,132,252,0.04); }
.glass-card-coral { border-color: rgba(251,113,133,0.22); background: rgba(251,113,133,0.04); }

/* ── Badge ── */
.badge {
    display: inline-block;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.5px;
    padding: 2px 8px;
    border-radius: 20px;
    font-family: 'JetBrains Mono', monospace;
    text-transform: lowercase;
}
.badge-teal   { background: rgba(15,242,200,0.15);  color: #0ff2c8; border: 1px solid rgba(15,242,200,0.3); }
.badge-blue   { background: rgba(91,142,240,0.15);  color: #8ab4f8; border: 1px solid rgba(91,142,240,0.3); }
.badge-amber  { background: rgba(245,197,66,0.15);  color: #f5c542; border: 1px solid rgba(245,197,66,0.3); }
.badge-purple { background: rgba(192,132,252,0.15); color: #c084fc; border: 1px solid rgba(192,132,252,0.3); }
.badge-coral  { background: rgba(251,113,133,0.15); color: #fb7185; border: 1px solid rgba(251,113,133,0.3); }

/* ── Topbar ── */
.gov-topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 24px;
    margin-bottom: 20px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    backdrop-filter: blur(20px);
    position: relative;
    overflow: hidden;
}
.gov-topbar::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(15,242,200,0.4), rgba(91,142,240,0.4), transparent);
}
.gov-logo-text {
    font-size: 20px;
    font-weight: 700;
    background: linear-gradient(135deg, #0ff2c8, #5b8ef0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.5px;
}
.gov-session {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: rgba(255,255,255,0.35);
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 5px 12px;
}

/* ── Pipeline step ── */
.pipeline-step {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 7px 10px;
    border-radius: 8px;
    margin-bottom: 4px;
    font-size: 12px;
}
.pipeline-step.done    { background: rgba(15,242,200,0.08);  color: #0ff2c8; }
.pipeline-step.active  { background: rgba(245,197,66,0.10); color: #f5c542; }
.pipeline-step.pending { color: rgba(255,255,255,0.30); }
.step-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}
.dot-done    { background: #0ff2c8; box-shadow: 0 0 6px rgba(15,242,200,0.6); }
.dot-active  { background: #f5c542; box-shadow: 0 0 6px rgba(245,197,66,0.6); animation: pulse 1s ease-in-out infinite; }
.dot-pending { background: rgba(255,255,255,0.18); }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

/* ── Domain pill ── */
.domain-pill {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 12px;
    border-radius: 10px;
    border: 1px solid transparent;
    cursor: pointer;
    margin-bottom: 4px;
    transition: all 0.15s;
    background: rgba(255,255,255,0.02);
}
.domain-pill:hover { border-color: rgba(255,255,255,0.12); background: rgba(255,255,255,0.05); }
.domain-pill.active-domain { border-color: rgba(15,242,200,0.3); background: rgba(15,242,200,0.07); }
.domain-count-pill {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    background: rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 1px 7px;
    color: rgba(255,255,255,0.45);
}

/* ── Checklist row ── */
.checklist-row {
    display: grid;
    grid-template-columns: 24px 1fr 110px 80px;
    gap: 12px;
    align-items: center;
    padding: 11px 14px;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    transition: background 0.12s;
    font-size: 12.5px;
}
.checklist-row:hover { background: rgba(255,255,255,0.03); }
.checklist-row.header {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.6px;
    color: rgba(255,255,255,0.3);
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    padding: 8px 14px;
}

/* ── Chunk card ── */
.chunk-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 12px 14px;
    margin-bottom: 8px;
    transition: border-color 0.15s;
}
.chunk-card:hover { border-color: rgba(255,255,255,0.16); }
.chunk-meta {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 7px;
    flex-wrap: wrap;
}
.chunk-source {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: rgba(255,255,255,0.3);
}
.chunk-score {
    margin-left: auto;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #0ff2c8;
    font-weight: 500;
}
.chunk-text {
    font-size: 12px;
    color: rgba(255,255,255,0.55);
    line-height: 1.6;
    font-style: italic;
}

/* ── Sidebar labels ── */
.sidebar-label {
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: rgba(255,255,255,0.25) !important;
    margin: 16px 0 8px;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Status glow ── */
.status-glow {
    font-size: 10px;
    font-family: 'JetBrains Mono', monospace;
    color: #0ff2c8;
    background: rgba(15,242,200,0.10);
    border: 1px solid rgba(15,242,200,0.25);
    border-radius: 20px;
    padding: 3px 10px;
    display: inline-flex;
    align-items: center;
    gap: 5px;
}
.live-dot {
    width: 5px; height: 5px;
    border-radius: 50%;
    background: #0ff2c8;
    box-shadow: 0 0 5px rgba(15,242,200,0.8);
    display: inline-block;
    animation: pulse 1.5s ease-in-out infinite;
}

.col-label { color: rgba(255,255,255,0.40) !important; font-size: 11px !important; }
.mono { font-family: 'JetBrains Mono', monospace; }

/* ── Professional spacing + type hierarchy ── */
.block-container {
    padding-top: 1.1rem !important;
    padding-bottom: 1.2rem !important;
}
h1, h2, h3 {
    letter-spacing: -0.02em;
}
p, li {
    line-height: 1.45;
}

/* ── KPI strip ── */
.kpi-strip {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 10px;
    margin: 4px 0 10px;
}
.kpi-card {
    background: var(--glass-bg);
    border: 1px solid var(--glass-bdr);
    border-radius: 12px;
    padding: 12px 14px;
    backdrop-filter: blur(10px);
}
.kpi-label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.9px;
    color: var(--text-sec);
    margin-bottom: 4px;
    font-family: 'JetBrains Mono', monospace;
}
.kpi-value {
    font-size: 19px;
    font-weight: 700;
    color: var(--text-pri);
}
.kpi-sub {
    margin-top: 2px;
    font-size: 11px;
    color: var(--text-ter);
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())[:8]

if "checklist" not in st.session_state:
    st.session_state.checklist = []
    
if "doc_meta" not in st.session_state:
    st.session_state.doc_meta = {"name": "", "chunks": 0, "reqs": 0, "docs": []}

if "pipeline_progress" not in st.session_state:
    st.session_state.pipeline_progress = 0.0

if "active_domain" not in st.session_state:
    st.session_state.active_domain = "all"
    
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_user_query" not in st.session_state:
    st.session_state.last_user_query = ""
    
if "docs" not in st.session_state:
    st.session_state.docs = []    

# ── Topbar ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="gov-topbar">
  <div style="display:flex;align-items:center;gap:14px">
    <div style="width:36px;height:36px;background:linear-gradient(135deg,rgba(15,242,200,0.3),rgba(91,142,240,0.3));border:1px solid rgba(15,242,200,0.4);border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:18px">🏛️</div>
    <div>
      <div class="gov-logo-text">GovCheck AI</div>
      <div style="font-size:10px;color:rgba(255,255,255,0.3);letter-spacing:0.5px;margin-top:1px">Hybrid RAG · Compliance Intelligence</div>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:12px">
    <span class="status-glow"><span class="live-dot"></span>Session active</span>
    <span class="gov-session">{st.session_state.user_id}</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Functions ───────────────────────────────────────────────────────────────
def get_badge_info(domain):
    colors = ["badge-teal", "badge-blue", "badge-purple", "badge-amber", "badge-coral"]
    d_str = str(domain).lower()
    idx = sum(ord(c) for c in d_str) % len(colors)
    return colors[idx], d_str.title().replace("_", " ")


def checklist_rows_for_csv(checklist: list) -> tuple[list[dict], list[str]]:
    """Stable columns aligned with API checklist items + UI fields."""
    cols = [
        "id",
        "requirement",
        "domain",
        "source_section",
        "priority",
        "action_type",
        "evidence_required",
        "chunk_id",
        "source_url",
        "compliance_framework",
        "done",
        "rt_score",
        "rt_risk",
    ]
    rows = []
    for r in checklist:
        req = r.get("req") or r.get("item") or r.get("requirement") or ""
        src = (
            r.get("source_section")
            or r.get("sourceSection")
            or r.get("source")
            or ""
        )
        rows.append(
            {
                "id": r.get("id", ""),
                "requirement": req,
                "domain": r.get("domain", ""),
                "source_section": src,
                "priority": r.get("priority", ""),
                "action_type": r.get("action_type", ""),
                "evidence_required": r.get("evidence_required", ""),
                "chunk_id": r.get("chunk_id", ""),
                "source_url": r.get("source_url", ""),
                "compliance_framework": r.get("compliance_framework", ""),
                "done": r.get("done", ""),
                "rt_score": r.get("rt_score", ""),
                "rt_risk": r.get("rt_risk", ""),
            }
        )
    return rows, cols


def checklist_dataframe(checklist: list[dict]) -> pd.DataFrame:
    rows = []
    for i, r in enumerate(checklist, start=1):
        rows.append({
            "ID": r.get("id", f"{i:02d}"),
            "Requirement": r.get("req") or r.get("item") or r.get("requirement") or "",
            "Domain": r.get("domain", ""),
            "Priority": r.get("priority", "Medium"),
            "Action Type": r.get("action_type", "Process"),
            "Source Section": r.get("source_section") or r.get("source") or "N/A",
            "Evidence Required": r.get("evidence_required", ""),
            "Chunk ID": r.get("chunk_id", ""),
            "Retrieval Score": float(r.get("retrieval_score", 0.0) or 0.0),
            "Done": bool(r.get("done", False)),
        })
    return pd.DataFrame(rows)


def build_audit_report_markdown(checklist: list[dict], user_id: str) -> str:
    total = len(checklist)
    done = sum(1 for x in checklist if x.get("done", False))
    completion = int((done / total) * 100) if total else 0
    by_domain: dict[str, int] = {}
    by_priority: dict[str, int] = {"High": 0, "Medium": 0, "Low": 0}
    for item in checklist:
        domain = str(item.get("domain", "unclassified"))
        by_domain[domain] = by_domain.get(domain, 0) + 1
        p = str(item.get("priority", "Medium")).capitalize()
        if p not in by_priority:
            by_priority[p] = 0
        by_priority[p] += 1

    lines = [
        "# GovCheck Audit Report",
        "",
        "## Executive Summary",
        f"- Session ID: `{user_id}`",
        f"- Total checklist items: **{total}**",
        f"- Items marked complete: **{done}**",
        f"- Completion rate: **{completion}%**",
        "",
        "## Priority Distribution",
        f"- High: {by_priority.get('High', 0)}",
        f"- Medium: {by_priority.get('Medium', 0)}",
        f"- Low: {by_priority.get('Low', 0)}",
        "",
        "## Domain Coverage",
    ]
    for domain, cnt in sorted(by_domain.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"- {domain}: {cnt}")
    lines += ["", "## Detailed Checklist", ""]

    for i, item in enumerate(checklist, start=1):
        lines.extend([
            f"### {i}. {item.get('item', item.get('req', 'Requirement'))}",
            f"- Domain: {item.get('domain', 'N/A')}",
            f"- Priority: {item.get('priority', 'Medium')}",
            f"- Action Type: {item.get('action_type', 'Process')}",
            f"- Source Section: {item.get('source_section', item.get('source', 'N/A'))}",
            f"- Evidence Required: {item.get('evidence_required', 'N/A')}",
            f"- Chunk ID: {item.get('chunk_id', 'N/A')}",
            "",
        ])
    return "\n".join(lines)


def api_post(path: str, **kwargs):
    kwargs.setdefault("timeout", HTTP_TIMEOUT_SEC)
    return requests.post(f"{API_URL}{path}", **kwargs)


def api_get(path: str, **kwargs):
    kwargs.setdefault("timeout", HTTP_TIMEOUT_SEC)
    return requests.get(f"{API_URL}{path}", **kwargs)


def _format_citations(text: str) -> str:
    # Render [chunk_id:abc] as compact citation chips.
    return re.sub(
        r"\[chunk_id:([^\]]+)\]",
        r"<span class='badge badge-blue'>chunk:\1</span>",
        text,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    sc1, sc2 = st.columns([4, 1])
    with sc1:
        st.markdown('<div style="padding:16px 0 8px"><div class="gov-logo-text" style="font-size:15px">⬆ Ingestion</div></div>', unsafe_allow_html=True)
    with sc2:
        st.write("") # small padding
        if st.button("☀️" if st.session_state.ui_theme == "dark" else "🌙", key="theme_toggle"):
            st.session_state.ui_theme = "light" if st.session_state.ui_theme == "dark" else "dark"
            st.rerun()

    uploaded = st.file_uploader(
        "Upload governance documents",
        type=["pdf", "docx", "xlsx", "csv", "txt"],
        accept_multiple_files=False,
        label_visibility="collapsed",
        help="Supports PDF, Word, Excel, CSV, TXT",
    )

    st.markdown("""
    <div style="display:flex;flex-wrap:wrap;gap:5px;margin-top:8px;margin-bottom:4px">
      <span class="badge badge-teal">.pdf</span>
      <span class="badge badge-blue">.docx</span>
      <span class="badge badge-amber">.xlsx</span>
      <span class="badge badge-purple">.csv</span>
      <span class="badge" style="background:rgba(255,255,255,0.06);color:rgba(255,255,255,0.4);border:1px solid rgba(255,255,255,0.12)">.txt</span>
      <span class="badge" style="background:rgba(255,255,255,0.06);color:rgba(255,255,255,0.4);border:1px solid rgba(255,255,255,0.12)">URL</span>
    </div>
    """, unsafe_allow_html=True)

    url_input = st.text_input("Or paste a URL / Web link", placeholder="https://...", label_visibility="visible")

    if uploaded or url_input:
        process_btn = st.button("▶  Process documents", type="primary", use_container_width=True, key="process_btn")
        
        if process_btn:
            with st.spinner("Processing..."):
                if uploaded:
                    files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                    data = {"user_id": st.session_state["user_id"]}
                    st.session_state.doc_meta["name"] = uploaded.name
                    res = api_post("/api/upload", files=files, data=data)
                    st.session_state.docs.append((uploaded.name, uploaded.type, "uploaded"))
                elif url_input:
                    data = {"link": url_input.strip(), "user_id": st.session_state["user_id"]}
                    st.session_state.doc_meta["name"] = str(url_input).split("/")[-1][:30] or "Cloud Document"
                    res = api_post("/api/upload", data=data)
                    st.session_state.docs.append((url_input, "URL", "link"))
                
                if res.status_code == 200:
                    job_id = res.json()["job_id"]
                    prog_bar = st.progress(0.0, text="")
                    
                    db_completed = False
                    poll_error = None
                    poll_ms = float(os.getenv("STREAMLIT_STATUS_POLL_SEC", "0.35"))
                    while not db_completed:
                        time.sleep(poll_ms)
                        try:
                            status_res = api_get(f"/api/status/{job_id}")
                            if status_res.status_code != 200:
                                poll_error = f"Status HTTP {status_res.status_code}"
                                break
                            status_data = status_res.json()
                            st.session_state.pipeline_progress = status_data["progress"] / 100.0
                            prog_bar.progress(st.session_state.pipeline_progress)
                            if status_data["status"] == "error":
                                poll_error = status_data.get("message", "Pipeline failed")
                                break
                            if status_data["status"] == "completed" or status_data["progress"] == 100:
                                db_completed = True
                        except Exception as ex:
                            poll_error = str(ex)
                            break

                    if poll_error:
                        st.error(f"Ingestion failed: {poll_error}")
                    else:
                        st.success("Pipeline complete!", icon="✅")

                    # Fetch initial checklist
                    if not poll_error:
                        payload_data = {
                            "query": "Extract all compliance policies, requirements, and checklist items. Strictly format as JSON.",
                            "domain": "all",
                            "user_id": st.session_state["user_id"],
                        }
                        chat_res = api_post("/api/chat", json=payload_data)
                        if chat_res.status_code == 200:
                            payload = chat_res.json()
                            raw_data = payload.get("raw_data")
                            st.session_state.checklist = raw_data if raw_data is not None else []
                            if not st.session_state.checklist:
                                st.warning("No checklist items were extracted from this source.")

                            st.session_state.doc_meta["reqs"] = len(st.session_state.checklist)
                            # Re-format checklist to match the new UI's expected format if needed
                            for i, c in enumerate(st.session_state.checklist):
                                if "id" not in c:
                                    c["id"] = f"{i+1:02d}"
                                if "req" not in c:
                                    c["req"] = c.get("requirement", c.get("item", "Unknown requirement"))
                                if "done" not in c:
                                    c["done"] = False
                                domain_str = str(c.get("domain", "")).lower().replace(" ", "_")
                                if "badge" not in c:
                                    c["badge"], _ = get_badge_info(domain_str)
                                if "source" not in c:
                                    c["source"] = c.get(
                                        "sourceSection",
                                        c.get("source_section", c.get("source", "N/A")),
                                    )
                        else:
                            st.session_state.checklist = []
                            st.error(f"Checklist extraction failed (HTTP {chat_res.status_code}).")
                else:
                    st.error(f"Error {res.status_code}: {res.text}")

    # Pipeline status
    st.markdown('<div class="sidebar-label">ETL pipeline</div>', unsafe_allow_html=True)
    progress = st.session_state.pipeline_progress

    steps = [
        ("Extracting text",       1.0 if progress > 0 else 0.0),
        ("Chunking & metadata",   1.0 if progress >= 0.25 else 0.0),
        ("Domain classification", 1.0 if progress >= 0.50 else 0.0),
        ("Embedding vectors",     1.0 if progress >= 0.75 else (progress if progress > 0.50 else 0.0)),
        ("BM25 indexing",         1.0 if progress >= 1.0 else 0.0),
    ]

    for label, done_frac in steps:
        if done_frac >= 1.0:
            state, dot_cls, cls = "done",    "dot-done",    "done"
        elif done_frac > 0.0:
            state, dot_cls, cls = "active",  "dot-active",  "active"
        else:
            state, dot_cls, cls = "pending", "dot-pending", "pending"
        st.markdown(f"""
        <div class="pipeline-step {cls}">
          <span class="step-dot {dot_cls}"></span>
          {label}
        </div>
        """, unsafe_allow_html=True)

    st.progress(min(progress, 1.0))
    chunks_done = int(progress * 458)  # Dummy approx number or actual
    st.markdown(f'<div style="font-size:10px;color:rgba(255,255,255,0.3);text-align:right;margin-top:2px;font-family:JetBrains Mono,monospace">{int(progress*100)}%</div>', unsafe_allow_html=True)

    # Domain filter
    st.markdown('<div class="sidebar-label">Filter by domain</div>', unsafe_allow_html=True)

    all_domains = list(set([x.get("domain", "general") for x in st.session_state.checklist]))
    
    # Calculate counts dynamically
    dom_counts = {"all": len(st.session_state.checklist)}
    for d in all_domains:
        dom_counts[d] = len([x for x in st.session_state.checklist if x.get("domain") == d])

    domains = [
        ("all",          "All domains",     "rgba(255,255,255,0.45)", dom_counts["all"]),
    ]
    
    hex_colors = ["#0ff2c8", "#5b8ef0", "#f5c542", "#c084fc", "#fb7185"]
    for d in all_domains:
        label = d.replace("_", " ").title()
        color = hex_colors[sum(ord(c) for c in d) % len(hex_colors)]
        domains.append((d, label, color, dom_counts[d]))

    for key, label, color, count in domains:
        active_cls = "active-domain" if st.session_state.active_domain == key else ""
        st.markdown(f"""
        <div class="domain-pill {active_cls}">
          <div style="display:flex;align-items:center;gap:8px">
            <span style="width:8px;height:8px;border-radius:2px;background:{color};display:inline-block;box-shadow:0 0 5px {color}55"></span>
            <span style="font-size:12px;color:rgba(255,255,255,0.75)">{label}</span>
          </div>
          <span class="domain-count-pill">{count}</span>
        </div>
        """, unsafe_allow_html=True)
        if st.button(label, key=f"dom_{key}", use_container_width=True):
            st.session_state.active_domain = key
            st.rerun()

    st.markdown('<div class="sidebar-label">Session</div>', unsafe_allow_html=True)
    if st.button(
        "⟳ Start Over",
        use_container_width=True,
        key="sidebar_start_over",
        help="Reset session: clear checklist, chat, uploads, and pipeline progress for a new document.",
    ):
        st.session_state.clear()
        st.rerun()

# ── Main content ──────────────────────────────────────────────────────────────
done_count = sum(1 for item in st.session_state.checklist if item.get("done", False))
total      = len(st.session_state.checklist)

pct = int(done_count / total * 100) if total > 0 else 0
scores = [float(x.get("retrieval_score", 0.0) or 0.0) for x in st.session_state.checklist if str(x.get("retrieval_score", "")) != ""]
avg_score = f"{(sum(scores) / max(1, len(scores))):.3f}" if scores else "N/A"
st.markdown(
    f"""
    <div class="kpi-strip">
      <div class="kpi-card"><div class="kpi-label">Ingested Docs</div><div class="kpi-value">{len(st.session_state.docs)}</div><div class="kpi-sub">Session scope</div></div>
      <div class="kpi-card"><div class="kpi-label">Checklist Items</div><div class="kpi-value">{total}</div><div class="kpi-sub">Extracted controls</div></div>
      <div class="kpi-card"><div class="kpi-label">Completion</div><div class="kpi-value">{done_count}/{total}</div><div class="kpi-sub">{pct}% marked complete</div></div>
      <div class="kpi-card"><div class="kpi-label">Avg Retrieval Score</div><div class="kpi-value">{avg_score}</div><div class="kpi-sub">Hybrid dense+sparse</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["📋  Checklist", "🔍  RAG Chat", "📊  Analytics"])

# ── TAB 1: Checklist ──────────────────────────────────────────────────────────
with tab1:
    # Export row
    ec1, ec2, ec3, ec4, ec5 = st.columns([1.5, 1.5, 1.5, 2.5, 2.0])
    
    has_data = len(st.session_state.checklist) > 0

    with ec1:
        if has_data:
            csv_rows, csv_cols = checklist_rows_for_csv(st.session_state.checklist)
            csv_data = pd.DataFrame(csv_rows, columns=csv_cols).to_csv(index=False).encode("utf-8")
        else:
            csv_data = b""
        st.download_button("⬇ CSV", data=csv_data, file_name="checklist.csv", mime="text/csv", use_container_width=True, disabled=not has_data, type="primary")
    with ec2:
        json_data = json.dumps(st.session_state.checklist, indent=2) if has_data else "{}"
        st.download_button("⬇ JSON", data=json_data, file_name="checklist.json", mime="application/json", use_container_width=True, disabled=not has_data, type="primary")
    with ec3:
        import io
        buf = io.BytesIO()
        if has_data:
            pd.DataFrame(st.session_state.checklist).to_excel(buf, index=False, engine='openpyxl')
        st.download_button("⬇ Excel", data=buf.getvalue(), file_name="checklist.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True, disabled=not has_data, type="primary")
    with ec4:
        report_md = build_audit_report_markdown(st.session_state.checklist, st.session_state.user_id) if has_data else ""
        st.download_button(
            "⬇ Audit Report",
            data=report_md,
            file_name="audit_report.md",
            mime="text/markdown",
            use_container_width=True,
            disabled=not has_data,
            type="primary",
            help="Executive-style audit summary and detailed checklist."
        )
    with ec5:
        if st.button("⟳ Start Over", type="primary", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    filtered_checklist = [c for c in st.session_state.checklist if st.session_state.active_domain == "all" or c.get("domain") == st.session_state.active_domain]
    sdf = checklist_dataframe(filtered_checklist)
    sort_col1, sort_col2, sort_col3 = st.columns([2.0, 1.4, 1.2])
    with sort_col1:
        sort_by = st.selectbox(
            "Sort by",
            options=["Priority", "Retrieval Score", "Domain", "ID"],
            index=0,
            key="sort_by_table",
        )
    with sort_col2:
        sort_order = st.selectbox(
            "Order",
            options=["Ascending", "Descending"],
            index=0 if sort_by in {"Priority", "Domain", "ID"} else 1,
            key="sort_order_table",
        )
    with sort_col3:
        st.caption(f"{len(sdf)} rows")

    if not sdf.empty:
        if sort_by == "Priority":
            priority_rank = {"High": 0, "Medium": 1, "Low": 2}
            sdf["_priority_rank"] = sdf["Priority"].map(priority_rank).fillna(1)
            sdf = sdf.sort_values(
                by="_priority_rank",
                ascending=(sort_order == "Ascending"),
                kind="stable",
            ).drop(columns=["_priority_rank"])
        else:
            sdf = sdf.sort_values(
                by=sort_by,
                ascending=(sort_order == "Ascending"),
                kind="stable",
            )

    st.dataframe(
        sdf,
        hide_index=True,
        use_container_width=True,
        height=480,
        column_config={
            "Requirement": st.column_config.TextColumn("Requirement", width="large"),
            "Domain": st.column_config.TextColumn("Domain", width="medium"),
            "Priority": st.column_config.TextColumn("Priority", width="small"),
            "Action Type": st.column_config.TextColumn("Action Type", width="small"),
            "Source Section": st.column_config.TextColumn("Source Section", width="medium"),
            "Evidence Required": st.column_config.TextColumn("Evidence Required", width="large"),
            "Chunk ID": st.column_config.TextColumn("Chunk ID", width="medium"),
            "Retrieval Score": st.column_config.NumberColumn("Retrieval Score", format="%.4f"),
            "Done": st.column_config.CheckboxColumn("Done"),
        },
    )

# ── TAB 2: RAG Query ──────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="glass-card glass-card-teal">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:11px;color:rgba(15,242,200,0.7);font-weight:600;letter-spacing:0.5px;margin-bottom:10px">HYBRID RAG INTERFACE</div>', unsafe_allow_html=True)

    t1, t2 = st.columns([1.2, 1.2])
    with t1:
        if st.button("↻ Regenerate last response", use_container_width=True, disabled=not bool(st.session_state.last_user_query)):
            if st.session_state.last_user_query:
                st.session_state["chat_history"].append({"role": "user", "content": st.session_state.last_user_query})
                st.rerun()
    with t2:
        if st.button("⟲ Retry last request", use_container_width=True, disabled=not bool(st.session_state.last_user_query)):
            if st.session_state.last_user_query:
                st.session_state["chat_history"].append({"role": "user", "content": st.session_state.last_user_query})
                st.rerun()

    chat_container = st.container(height=350)
    with chat_container:
        for msg in st.session_state["chat_history"]:
            role_emoji = "👤" if msg["role"] == "user" else "🤖"
            role_color = "#8ab4f8" if msg["role"] == "user" else "#0ff2c8"
            content = str(msg.get("content", ""))
            if msg["role"] == "assistant":
                content = _format_citations(content)
            st.markdown(f'<div style="margin-bottom:14px;"><strong style="color:{role_color};">{role_emoji} {msg["role"].title()}</strong><div style="font-size:13px; color:rgba(255,255,255,0.8); margin-top:4px;">{content}</div></div>', unsafe_allow_html=True)

    query = st.chat_input("Ask about the policies...")

    if query:
        st.session_state.last_user_query = query
        st.session_state["chat_history"].append({"role": "user", "content": query})
        st.rerun()

    # Process AI if last is user
    if st.session_state["chat_history"] and st.session_state["chat_history"][-1]["role"] == "user":
        query_text = st.session_state["chat_history"][-1]["content"]
        with chat_container:
            st.markdown(f'<div style="margin-bottom:8px;"><strong style="color:#0ff2c8;">🤖 Assistant</strong>', unsafe_allow_html=True)
            history_for_backend = st.session_state["chat_history"][:-1][-8:]
            payload_data = {
                "query": query_text,
                "domain": st.session_state["active_domain"],
                "user_id": st.session_state["user_id"],
                "chat_history": history_for_backend,
            }
            
            with st.spinner("Thinking..."):
                try:
                    def stream_res():
                        with requests.post(
                            f"{API_URL}/api/chat/stream",
                            json=payload_data,
                            stream=True,
                            timeout=(STREAM_CONNECT_TIMEOUT_SEC, STREAM_READ_TIMEOUT_SEC),
                        ) as response:
                            if response.status_code == 200:
                                for line in response.iter_content(chunk_size=1024, decode_unicode=True):
                                    if line:
                                        yield line
                            else:
                                yield "Backend stream error."
                    reply = st.write_stream(stream_res())
                    if "I cannot verify this from the provided documents." in str(reply):
                        st.warning("Weak evidence detected: response could not be strongly verified from retrieved documents.")
                    st.session_state["chat_history"].append({"role": "assistant", "content": reply})
                    st.rerun()
                except Exception as e:
                    st.error(f"Connection failed: {e}")
                    st.session_state["chat_history"].append({"role": "assistant", "content": f"Error: {e}"})
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ── TAB 3: Analytics ──────────────────────────────────────────────────────────
with tab3:
    if not st.session_state.checklist:
        st.info("No analytics yet. Process a document to generate checklist insights.")
    else:
        adf = checklist_dataframe(st.session_state.checklist)
        adf["Priority"] = adf["Priority"].fillna("Medium").astype(str).str.capitalize()
        adf["Domain"] = adf["Domain"].fillna("unclassified").astype(str)
        adf["Done"] = adf["Done"].astype(bool)
        adf["Has Evidence"] = adf["Evidence Required"].fillna("").astype(str).str.strip().str.len() > 0

        # Risk-weighted score (simple heuristic)
        risk_w = {"High": 3, "Medium": 2, "Low": 1}
        adf["Risk Weight"] = adf["Priority"].map(risk_w).fillna(2)
        total_risk = float(adf["Risk Weight"].sum())
        open_risk = float(adf.loc[~adf["Done"], "Risk Weight"].sum())
        risk_reduction_pct = int(((total_risk - open_risk) / total_risk) * 100) if total_risk > 0 else 0

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Open Items", int((~adf["Done"]).sum()))
        with k2:
            st.metric("High Priority Open", int(((adf["Priority"] == "High") & (~adf["Done"])).sum()))
        with k3:
            st.metric("Evidence Coverage", f"{int(adf['Has Evidence'].mean() * 100)}%")
        with k4:
            st.metric("Risk Reduction", f"{risk_reduction_pct}%")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Priority Distribution")
            pr = adf.groupby("Priority", as_index=False).size().rename(columns={"size": "Count"})
            pr["Priority"] = pd.Categorical(pr["Priority"], categories=["High", "Medium", "Low"], ordered=True)
            pr = pr.sort_values("Priority")
            st.bar_chart(pr.set_index("Priority")["Count"], use_container_width=True)

        with c2:
            st.markdown("##### Completion by Domain")
            dom = (
                adf.groupby("Domain", as_index=False)
                .agg(total=("ID", "count"), completed=("Done", "sum"))
            )
            dom["Completion %"] = ((dom["completed"] / dom["total"]) * 100).round(1)
            st.dataframe(
                dom.sort_values("Completion %", ascending=False),
                hide_index=True,
                use_container_width=True,
                height=250,
            )

        st.markdown("##### Pending Critical Actions")
        pending_critical = adf[(~adf["Done"]) & (adf["Priority"] == "High")][
            ["ID", "Requirement", "Domain", "Source Section", "Chunk ID"]
        ]
        if pending_critical.empty:
            st.success("No pending high-priority actions.")
        else:
            st.dataframe(
                pending_critical,
                hide_index=True,
                use_container_width=True,
                height=260,
            )

        analytics_export = adf.copy()
        analytics_export["Completion % by Domain"] = analytics_export["Domain"].map(
            dict(
                (d, v)
                for d, v in (
                    adf.groupby("Domain")
                    .apply(lambda x: round(float(x["Done"].mean() * 100), 1))
                    .to_dict()
                    .items()
                )
            )
        )
        st.download_button(
            "⬇ Export Analytics CSV",
            data=analytics_export.to_csv(index=False).encode("utf-8"),
            file_name="analytics_report.csv",
            mime="text/csv",
            use_container_width=False,
        )


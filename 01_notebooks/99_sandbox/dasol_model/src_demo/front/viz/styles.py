import streamlit as st


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

        :root { --bg: #f0f2f5; --text: #111827; --muted: #6b7280; --red: #e50914; }

        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        .stApp { background: #f0f2f5; color: var(--text); }
        .block-container { max-width: 1420px; padding-top: 4.8rem !important; padding-bottom: 2rem; }

        /* ── Top Navbar ── */
        .top-navbar { position: fixed; top: 0; left: 0; right: 0; z-index: 9999; background: #111827; display: flex; justify-content: space-between; align-items: center; padding: 0 28px; height: 52px; border-bottom: 1px solid rgba(255,255,255,0.07); }
        .navbar-left  { display: flex; align-items: center; gap: 10px; }
        .navbar-logo  { color: #E50914; font-size: 1.4rem; font-weight: 900; }
        .navbar-title { color: #fff; font-weight: 800; font-size: 0.97rem; }
        .navbar-right { display: flex; align-items: center; gap: 10px; }
        .navbar-btn-active { background: #E50914; color: white; padding: 6px 16px; border-radius: 8px; font-weight: 700; font-size: 0.88rem; }
        .navbar-btn   { color: #d1d5db; padding: 6px 16px; border: 1px solid #374151; border-radius: 8px; font-weight: 700; font-size: 0.88rem; }
        .navbar-date  { color: #9ca3af; font-size: 0.83rem; margin: 0 4px; }
        .navbar-avatar { background: #E50914; color: white; width: 32px; height: 32px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-weight: 900; font-size: 0.78rem; }

        /* ── 흰색 카드 박스 (st.container 대체) ── */
        .card-box {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 18px;
            padding: 22px 22px 18px 22px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.06);
            height: 100%;
        }
        /* 상단 카드 */
        .st-key-card-trend,
        .st-key-card-risk-donut {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 18px;
            padding: 22px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.06);
            height: 420px;
        }
        
        /* 하단 카드 */
        .st-key-card-ott-usage,
        .st-key-card-genre-chart,
        .st-key-card-churn-drivers {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 18px;
            padding: 22px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.06);
            height: 340px;
        }

        /* ── Hero ── */
        .hero-wrap { background: linear-gradient(135deg,#0f172a 0%,#1e293b 100%); border-radius: 18px; padding: 22px 28px; box-shadow: 0 8px 24px rgba(0,0,0,0.12); }
        .hero-title  { font-size: 1.6rem; font-weight: 900; color: white; margin-bottom: 4px; }
        .hero-sub    { color: #cbd5e1; font-size: 0.95rem; margin-bottom: 12px; }
        .hero-insight { color: #f8fafc; font-size: 0.9rem; background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.10); display: inline-block; padding: 7px 14px; border-radius: 999px; font-weight: 700; }

        /* ── Summary bar ── */
        .dashboard-summary-bar { padding: 11px 16px; border-radius: 12px; background: rgba(59,130,246,0.08); border: 1px solid rgba(59,130,246,0.20); color: #1d4ed8; font-size: 0.90rem; font-weight: 600; margin-bottom: 8px; }

        /* ── KPI cards ── */
        .kpi-card      { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 18px; padding: 20px 22px 18px 22px; min-height: 130px; box-shadow: 0 2px 12px rgba(0,0,0,0.06); }
        .kpi-top       { display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px; }
        .kpi-icon      { width: 42px; height: 42px; border-radius: 12px; display: flex; align-items: center; justify-content: center; background: #fef2f2; font-size: 1.15rem; }
        .kpi-delta-pos { color: #16a34a; font-size: 0.90rem; font-weight: 800; }
        .kpi-delta-neg { color: #dc2626; font-size: 0.90rem; font-weight: 800; }
        .kpi-label     { color: #6b7280; font-size: 0.92rem; margin-bottom: 6px; font-weight: 500; }
        .kpi-value     { color: #111827; font-size: 2rem; font-weight: 900; line-height: 1.1; }

        /* ── Section headings ── */
        .section-heading-wrap { margin-bottom: 12px; }
        .section-title { font-size: 1.1rem; font-weight: 800; color: #111827; margin-bottom: 3px; }
        .section-sub   { color: #9ca3af; font-size: 0.88rem; }

        /* ── Bars ── */
        .metric-row     { margin-bottom: 14px; }
        .metric-row-top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; font-size: 0.92rem; }
        .metric-label   { color: #374151; font-weight: 600; }
        .metric-value   { color: #111827; font-weight: 800; }
        .bar-track     { width: 100%; height: 10px; border-radius: 999px; background: #f1f5f9; overflow: hidden; }
        .bar-fill-blue { height: 100%; border-radius: 999px; background: linear-gradient(90deg,#3b82f6,#60a5fa); }
        .bar-fill-red  { height: 100%; border-radius: 999px; background: linear-gradient(90deg,#ef4444,#f87171); }

        /* ── Legend ── */
        .legend-item  { display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px; }
        .legend-left  { display: flex; align-items: center; gap: 8px; }
        .legend-right { display: flex; align-items: center; gap: 10px; min-width: 80px; justify-content: flex-end; }
        .legend-dot   { width: 11px; height: 11px; border-radius: 50%; flex-shrink: 0; }
        .legend-label { color: #374151; font-weight: 600; font-size: 0.92rem; }
        .legend-value { color: #111827; font-weight: 800; font-size: 0.92rem; }
        .legend-pct   { color: #9ca3af; font-weight: 600; font-size: 0.85rem; min-width: 36px; text-align: right; }

        /* ── Alert ── */
        .alert-box { margin-top: 16px; background: #fffbeb; border: 1px solid #fcd34d; color: #92400e; border-radius: 12px; padding: 10px 14px; font-size: 0.88rem; font-weight: 600; }

        /* ── User cards ── */
        .user-card { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 16px; padding: 16px 18px; margin-bottom: 12px; box-shadow: 0 1px 6px rgba(0,0,0,0.04); }
        .user-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; margin-bottom: 12px; flex-wrap: wrap; }
        .user-head-left  { display: flex; flex-direction: column; gap: 4px; }
        .user-name       { color: #111827; font-size: 1.02rem; font-weight: 800; }
        .user-subline    { color: #9ca3af; font-size: 0.85rem; font-weight: 500; }
        .user-pill       { background: #fef2f2; color: #dc2626; border: 1px solid #fecaca; border-radius: 999px; padding: 5px 12px; font-size: 0.80rem; font-weight: 800; white-space: nowrap; }
        .user-grid       { display: grid; grid-template-columns: repeat(4,minmax(0,1fr)); gap: 10px; }
        .user-meta       { background: #f9fafb; border-radius: 12px; padding: 10px 12px; border: 1px solid #f3f4f6; }
        .highlight-meta  { background: #f0f9ff; border: 1px solid #bae6fd; }
        .user-meta-label { color: #9ca3af; font-size: 0.78rem; margin-bottom: 4px; font-weight: 500; }
        .user-meta-value { color: #111827; font-size: 0.95rem; font-weight: 800; word-break: break-word; }

        /* ── Streamlit overrides ── */
        div[data-baseweb="select"] > div,
        .stFileUploader > div > div { background: #ffffff !important; border: 1px solid #e5e7eb !important; border-radius: 12px !important; color: #111827 !important; }
        .stMultiSelect [data-baseweb="tag"] { background: #e50914 !important; color: white !important; border-radius: 8px !important; }
        .stExpander { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 14px; margin-bottom: 10px; }
        .stPlotlyChart { margin-bottom: 4px; }
        label, .stMarkdown p { color: var(--text); }
        div.stButton > button       { background-color: #1f2937; color: white; border-radius: 10px; border: none; font-weight: 600; }
        div.stButton > button:hover { background-color: #374151; }

        /* ── Landing ── */
        .landing-wrap  { text-align: center; padding: 28px 0 12px 0; }
        .landing-badge { display: inline-block; padding: 6px 12px; border-radius: 999px; background: rgba(229,9,20,0.08); border: 1px solid rgba(229,9,20,0.20); color: #dc2626; font-size: 0.82rem; font-weight: 800; margin-bottom: 16px; }
        .landing-title { font-size: 2rem; font-weight: 900; color: #111827; margin-bottom: 8px; }
        .landing-sub   { color: #6b7280; font-size: 1rem; line-height: 1.6; max-width: 760px; margin: 0 auto; }
        .upload-card       { margin: 12px 0; padding: 24px; border-radius: 20px; background: #ffffff; border: 1px solid #e5e7eb; box-shadow: 0 4px 16px rgba(0,0,0,0.06); }
        .upload-card-title { color: #111827; font-size: 1.25rem; font-weight: 900; margin-bottom: 6px; }
        .upload-card-sub   { color: #6b7280; font-size: 0.92rem; margin-bottom: 8px; }

        @media (max-width: 900px) { .user-grid { grid-template-columns: 1fr 1fr; } }
        @media (max-width: 640px) {
            .user-grid { grid-template-columns: 1fr; }
            .hero-title, .landing-title { font-size: 1.35rem; }
            .top-navbar { padding: 0 14px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

import streamlit as st


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #07111f;
            --panel: #0f1a31;
            --panel-2: #16213d;
            --border: rgba(255,255,255,0.08);
            --text: #f9fafb;
            --muted: #9ca3af;
            --red: #e50914;
            --orange: #f97316;
            --yellow: #facc15;
            --green: #22c55e;
            --blue: #3b82f6;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(229,9,20,0.10), transparent 20%),
                linear-gradient(180deg, #07111f 0%, #08152a 100%);
            color: var(--text);
        }

        .block-container {
            max-width: 1420px;
            padding-top: 3.2rem;
            padding-bottom: 2rem;
        }

        .hero-wrap {
            background: linear-gradient(90deg, #05070d 0%, #08152a 100%);
            border: 1px solid var(--border);
            border-radius: 22px;
            padding: 20px 28px;
            box-shadow: 0 18px 40px rgba(0,0,0,0.28);
        }

        .hero-title {
            font-size: 1.75rem;
            font-weight: 900;
            color: white;
            margin-bottom: 4px;
        }

        .hero-sub {
            color: #cbd5e1;
            font-size: 0.98rem;
            margin-bottom: 10px;
        }

        .hero-insight {
            color: #f8fafc;
            font-size: 0.93rem;
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.06);
            display: inline-block;
            padding: 8px 12px;
            border-radius: 999px;
            font-weight: 700;
        }

        .landing-wrap {
            text-align: center;
            padding: 28px 0 12px 0;
        }

        .landing-badge {
            display: inline-block;
            padding: 6px 12px;
            border-radius: 999px;
            background: rgba(229,9,20,0.12);
            border: 1px solid rgba(229,9,20,0.28);
            color: #fecaca;
            font-size: 0.82rem;
            font-weight: 800;
            margin-bottom: 16px;
        }

        .landing-title {
            font-size: 2.1rem;
            font-weight: 900;
            color: #ffffff;
            margin-bottom: 8px;
        }

        .landing-sub {
            color: #cbd5e1;
            font-size: 1.02rem;
            line-height: 1.6;
            max-width: 760px;
            margin: 0 auto;
        }

        .upload-card {
            margin-top: 12px;
            margin-bottom: 12px;
            padding: 24px 24px 16px 24px;
            border-radius: 22px;
            background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 12px 28px rgba(0,0,0,0.18);
        }

        .upload-card-title {
            color: white;
            font-size: 1.3rem;
            font-weight: 900;
            margin-bottom: 6px;
        }

        .upload-card-sub {
            color: #9fb1c7;
            font-size: 0.94rem;
            margin-bottom: 8px;
        }

        .upload-guide {
            margin-top: 18px;
            padding: 16px 18px;
            border-radius: 18px;
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.06);
            text-align: left;
        }

        .upload-guide-title {
            color: white;
            font-size: 0.98rem;
            font-weight: 800;
            margin-bottom: 8px;
        }

        .upload-guide ul {
            margin: 0;
            padding-left: 18px;
            color: #cbd5e1;
            line-height: 1.8;
        }

        .dashboard-summary-bar {
            padding: 12px 14px;
            border-radius: 14px;
            background: rgba(59,130,246,0.10);
            border: 1px solid rgba(59,130,246,0.24);
            color: #dbeafe;
            font-size: 0.92rem;
            font-weight: 700;
            margin-bottom: 8px;
        }

        .kpi-card {
            background: linear-gradient(180deg, rgba(255,255,255,0.045), rgba(255,255,255,0.02));
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 18px 20px 16px 20px;
            min-height: 132px;
            box-shadow: 0 12px 28px rgba(0,0,0,0.18);
        }

        .kpi-top {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 10px;
        }

        .kpi-icon {
            width: 42px;
            height: 42px;
            border-radius: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: rgba(229,9,20,0.10);
            font-size: 1.1rem;
        }

        .kpi-delta-pos {
            color: #22c55e;
            font-size: 0.92rem;
            font-weight: 800;
        }

        .kpi-delta-neg {
            color: #f87171;
            font-size: 0.92rem;
            font-weight: 800;
        }

        .kpi-label {
            color: #cbd5e1;
            font-size: 0.94rem;
            margin-bottom: 8px;
        }

        .kpi-value {
            color: #ffffff;
            font-size: 2rem;
            font-weight: 900;
            line-height: 1.1;
        }

        .section-heading-wrap {
            margin-bottom: 10px;
        }

        .section-title {
            font-size: 1.25rem;
            font-weight: 900;
            color: white;
            margin-bottom: 4px;
        }

        .section-sub {
            color: var(--muted);
            font-size: 0.96rem;
            margin-bottom: 0;
        }

        .metric-row {
            margin-bottom: 16px;
        }

        .metric-row-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 7px;
            font-size: 0.96rem;
        }

        .metric-label {
            color: #f3f4f6;
            font-weight: 800;
        }

        .metric-value {
            color: #ffffff;
            font-weight: 900;
        }

        .bar-track {
            width: 100%;
            height: 12px;
            border-radius: 999px;
            background: rgba(255,255,255,0.09);
            overflow: hidden;
        }

        .bar-fill-blue {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #2563eb, #60a5fa);
        }

        .bar-fill-red {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #ef4444, #f87171);
        }

        .legend-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 10px;
            margin-bottom: 12px;
        }

        .legend-left {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .legend-right {
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .legend-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }

        .legend-label {
            color: #f3f4f6;
            font-weight: 800;
        }

        .legend-value {
            color: #ffffff;
            font-weight: 900;
        }

        .legend-pct {
            color: #94a3b8;
            font-weight: 700;
            font-size: 0.88rem;
        }

        .alert-box {
            margin-top: 18px;
            background: rgba(250, 204, 21, 0.10);
            border: 1px solid rgba(250, 204, 21, 0.45);
            color: #fef3c7;
            border-radius: 16px;
            padding: 12px 14px;
            font-size: 0.93rem;
            font-weight: 700;
        }

        .user-card {
            background: rgba(255,255,255,0.03);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 14px 16px;
            margin-bottom: 12px;
        }

        .user-head {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 12px;
            margin-bottom: 12px;
            flex-wrap: wrap;
        }

        .user-head-left {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }

        .user-name {
            color: white;
            font-size: 1.05rem;
            font-weight: 900;
        }

        .user-subline {
            color: #94a3b8;
            font-size: 0.88rem;
            font-weight: 700;
        }

        .user-pill {
            background: rgba(229,9,20,0.14);
            color: #fecaca;
            border: 1px solid rgba(229,9,20,0.35);
            border-radius: 999px;
            padding: 6px 11px;
            font-size: 0.82rem;
            font-weight: 900;
            white-space: nowrap;
        }

        .user-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 10px;
        }

        .user-meta {
            background: rgba(255,255,255,0.03);
            border-radius: 14px;
            padding: 12px;
        }

        .highlight-meta {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.05);
        }

        .user-meta-label {
            color: var(--muted);
            font-size: 0.82rem;
            margin-bottom: 6px;
        }

        .user-meta-value {
            color: white;
            font-size: 0.98rem;
            font-weight: 800;
            word-break: break-word;
        }

        div[data-baseweb="select"] > div,
        .stFileUploader > div > div {
            background: rgba(255,255,255,0.03) !important;
            border: 1px solid rgba(255,255,255,0.08) !important;
            border-radius: 14px !important;
        }

        .stMultiSelect [data-baseweb="tag"] {
            background: #e50914 !important;
            color: white !important;
            border-radius: 10px !important;
        }

        .stExpander {
            background: rgba(255,255,255,0.02);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 16px;
            margin-bottom: 10px;
        }

        .stPlotlyChart {
            margin-bottom: 6px;
        }

        label, .stMarkdown p {
            color: var(--text);
        }

        @media (max-width: 900px) {
            .user-grid {
                grid-template-columns: 1fr 1fr;
            }
        }

        @media (max-width: 640px) {
            .user-grid {
                grid-template-columns: 1fr;
            }

            .hero-title,
            .landing-title {
                font-size: 1.4rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
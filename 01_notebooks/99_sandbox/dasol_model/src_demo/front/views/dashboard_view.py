import streamlit as st
import pandas as pd
from textwrap import dedent

from ..state.state import (
    clear_uploaded_file, go_home,
    set_model_results, set_prediction_error, set_selected_model,
)
from ..ui.components import (
    render_churn_drivers, render_genre_chart, render_header,
    render_high_risk_users, render_kpi_card, render_ott_usage,
    render_risk_donut, render_section_heading, render_trend_card,
)
from ..ui.data import (
    CHURN_DRIVERS, GENRES, HIGH_RISK_USERS, KPI_DATA,
    MONTHLY_TREND, NAVBAR_UPDATE, OTT_USAGE, RISK_SEGMENTS,
    make_sample_dataframe,
)
from ..viz.charts import make_trend_chart
from ...model.predictor import (
    build_churn_drivers, build_genres, build_high_risk_users,
    build_kpi, build_monthly_trend, build_ott_usage,
    build_risk_segments, get_feature_names, models_available, predict_churn,
)


def _predict_sample_no_model(df: pd.DataFrame) -> pd.DataFrame:
    import numpy as np
    from ...model.predictor import RISK_BINS, RISK_LABELS
    df = df.copy()
    inactive  = df.get("InactiveDays",     pd.Series([30]*len(df))).fillna(30) / 60
    watch     = df.get("WeeklyWatchHours", pd.Series([8]*len(df))).fillna(8)
    watch_n   = 1 - (watch / (watch.max() + 1e-9))
    login     = df.get("LoginFrequency",   pd.Series([15]*len(df))).fillna(15) / 30
    diversity = df.get("ContentDiversity", pd.Series([5]*len(df))).fillna(5) / 10
    prob = 0.40*inactive + 0.25*watch_n + 0.20*(1-login) + 0.15*(1-diversity)
    df["churn_prob"] = np.clip(prob.values, 0.01, 0.99)
    df["risk_label"] = pd.cut(df["churn_prob"], bins=RISK_BINS, labels=RISK_LABELS, right=False).astype(str)
    return df


def _load_dataframe() -> pd.DataFrame | None:
    state = st.session_state
    if state.get("is_sample_mode"):
        return make_sample_dataframe()
    uf = state.get("uploaded_file")
    if uf is None:
        return None
    try:
        uf.seek(0)
        name = getattr(uf, "name", "")
        return pd.read_excel(uf) if name.endswith(".xlsx") else pd.read_csv(uf)
    except Exception as e:
        set_prediction_error(f"파일 읽기 오류: {e}")
        return None


def _run_prediction(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    return predict_churn(df, model_name) if model_name in models_available() else _predict_sample_no_model(df)


def render_dashboard_view() -> None:
    model_options  = ["XGBoost", "Random Forest", "Logistic Regression"]
    selected_model = st.session_state.get("selected_model", "XGBoost")

    # ── Navbar ───────────────────────────────────────────────────────────
    st.markdown(dedent(f"""
        <div class="top-navbar">
            <div class="navbar-left">
                <span class="navbar-logo">■</span>
                <span class="navbar-title">Netflix 이탈 예측 분석 대시보드</span>
            </div>
            <div class="navbar-right">
                <span class="navbar-btn-active">대시보드</span>
                <span class="navbar-btn">데이터 업로드</span>
                <span class="navbar-date">최종 업데이트: {NAVBAR_UPDATE}</span>
                <span class="navbar-avatar">NF</span>
            </div>
        </div>
    """), unsafe_allow_html=True)

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # ── Hero + 버튼 + 모델 선택 ──────────────────────────────────────────
    top_left, top_right = st.columns([1.5, 1])
    with top_left:
        render_header()
    with top_right:
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        if st.session_state.get("uploaded_file_name"):
            st.info(f"현재 데이터: {st.session_state.uploaded_file_name}")
        b1, b2 = st.columns(2)
        with b1:
            if st.button("← 메인으로", use_container_width=True):
                go_home(); st.rerun()
        with b2:
            if st.button("다시 업로드", use_container_width=True):
                clear_uploaded_file(); go_home(); st.rerun()
        new_model = st.selectbox(
            "🤖 예측 모델", options=model_options,
            index=model_options.index(selected_model),
            key="model_selector_dashboard",
        )
        if new_model != selected_model:
            set_selected_model(new_model); st.rerun()

    st.markdown(
        '<div class="dashboard-summary-bar">현재 화면은 업로드된 데이터를 기준으로 생성된 이탈 예측 분석 결과입니다.</div>',
        unsafe_allow_html=True,
    )

    # ── 예측 실행 ────────────────────────────────────────────────────────
    result_df = st.session_state.get("model_results")
    if result_df is None:
        df = _load_dataframe()
        if df is None:
            st.error("데이터를 불러올 수 없습니다. 메인 화면에서 파일을 다시 업로드해 주세요.")
            return
        with st.spinner(f"🔍 {selected_model} 모델로 이탈 예측 중..."):
            try:
                result_df = _run_prediction(df, selected_model)
                set_model_results(result_df)
            except Exception as e:
                set_prediction_error(str(e))
                st.error(f"예측 오류: {e}")
                _render_fallback()
                return

    # ── 집계 ─────────────────────────────────────────────────────────────
    feature_names   = get_feature_names(result_df)
    kpi_data        = build_kpi(result_df)
    risk_segments   = build_risk_segments(result_df)
    churn_drivers   = build_churn_drivers(selected_model, feature_names)
    ott_usage       = build_ott_usage(result_df)
    genres          = build_genres(result_df)
    monthly_trend   = build_monthly_trend(result_df)
    high_risk_users = build_high_risk_users(result_df, top_n=5)

    # 모델 배지
    if selected_model in models_available():
        st.markdown(f"<div style='font-size:12px;color:#6b7280;margin-bottom:8px'>📊 <b>{selected_model}</b> 예측 결과 | 총 {len(result_df):,}명 분석 완료</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='font-size:12px;color:#f97316;margin-bottom:8px'>⚠ 모델 파일 없음 — 내장 규칙 기반 예측으로 표시 중</div>", unsafe_allow_html=True)

    # ── KPI 카드 ─────────────────────────────────────────────────────────
    kpi_cols = st.columns(3)
    for col, item in zip(kpi_cols, kpi_data):
        with col:
            render_kpi_card(
                title=item.get("title","-"), value=item.get("value","-"),
                delta=item.get("delta","0%"), delta_type=item.get("delta_type","positive"),
                icon=item.get("icon","📌"),
            )

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # ── 추세 차트 + 위험도 도넛 ──────────────────────────────────────────
    left, right = st.columns([1.7, 1])
    with left:
        render_trend_card(
            "월별 이탈 추세",
            f"예측 확률 기반 이탈 추이 ({selected_model})",
            make_trend_chart(monthly_trend),
        )
    with right:
        render_risk_donut(risk_segments)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── OTT / 장르 / 이탈 요인 ───────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        render_ott_usage(ott_usage)
    with col2:
        render_genre_chart(genres)
    with col3:
        render_churn_drivers(churn_drivers)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # ── 고위험 사용자 ─────────────────────────────────────────────────────
    render_high_risk_users(high_risk_users)

    # ── CSV 다운로드 ──────────────────────────────────────────────────────
    # st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    # with st.expander("📥 예측 결과 다운로드"):
    #     dl_cols = ["churn_prob","risk_label"] + [c for c in result_df.columns if c not in ["churn_prob","risk_label"]]
    #     dl_df   = result_df[[c for c in dl_cols if c in result_df.columns]]
    #     st.download_button(
    #         label="CSV로 다운로드",
    #         data=dl_df.to_csv(index=False).encode("utf-8-sig"),
    #         file_name="churn_prediction_result.csv",
    #         mime="text/csv",
    #         use_container_width=True,
    #     )
    #     st.dataframe(dl_df.head(10), use_container_width=True)


def _render_fallback() -> None:
    kpi_cols = st.columns(3)
    for col, item in zip(kpi_cols, KPI_DATA):
        with col:
            render_kpi_card(**{k: item[k] for k in ("title","value","delta","delta_type","icon")})
    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    left, right = st.columns([1.7, 1])
    with left:
        render_trend_card("월별 이탈 추세", "샘플 데이터 기준", make_trend_chart(MONTHLY_TREND))
    with right:
        render_risk_donut(RISK_SEGMENTS)
    col1, col2, col3 = st.columns(3)
    with col1: render_ott_usage(OTT_USAGE)
    with col2: render_genre_chart(GENRES)
    with col3: render_churn_drivers(CHURN_DRIVERS)
    render_high_risk_users(HIGH_RISK_USERS)

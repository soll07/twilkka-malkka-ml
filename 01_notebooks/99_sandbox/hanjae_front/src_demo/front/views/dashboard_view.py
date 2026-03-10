import streamlit as st

from ..state.state import clear_uploaded_file, go_home
from ..ui.components import (
    render_churn_drivers,
    render_genre_chart,
    render_header,
    render_high_risk_users,
    render_kpi_card,
    render_ott_usage,
    render_risk_donut,
    render_section_heading,
)
from ..ui.data import (
    CHURN_DRIVERS,
    GENRES,
    HIGH_RISK_USERS,
    KPI_DATA,
    MONTHLY_TREND,
    OTT_USAGE,
    RISK_SEGMENTS,
)
from ..viz.charts import make_trend_chart


def render_dashboard_view() -> None:
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    top_left, top_right = st.columns([1.5, 1])

    with top_left:
        render_header()

    with top_right:
        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

        if st.session_state.get("uploaded_file_name"):
            st.info(f"현재 데이터: {st.session_state.uploaded_file_name}")

        button_col1, button_col2 = st.columns(2)

        with button_col1:
            if st.button("← 메인으로", use_container_width=True):
                go_home()
                st.rerun()

        with button_col2:
            if st.button("다시 업로드", use_container_width=True):
                clear_uploaded_file()
                go_home()
                st.rerun()

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="dashboard-summary-bar">
            현재 화면은 업로드된 데이터를 기준으로 생성된 이탈 예측 분석 결과입니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    kpi_cols = st.columns(3)
    for col, item in zip(kpi_cols, KPI_DATA):
        with col:
            render_kpi_card(
                title=item.get("title", "-"),
                value=item.get("value", "-"),
                delta=item.get("delta", "0.0%"),
                delta_type=item.get("delta_type", "positive"),
                icon=item.get("icon", "📌"),
            )

    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)

    left, right = st.columns([1.7, 1])

    with left:
        render_section_heading("📈 월별 이탈 추세", "최근 기간 동안의 이탈률과 시청 감소율 변화 (단위: %)")
        fig = make_trend_chart(MONTHLY_TREND)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        render_risk_donut(RISK_SEGMENTS)

    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        render_ott_usage(OTT_USAGE)

    with col2:
        render_genre_chart(GENRES)

    with col3:
        render_churn_drivers(CHURN_DRIVERS)

    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    render_high_risk_users(HIGH_RISK_USERS)
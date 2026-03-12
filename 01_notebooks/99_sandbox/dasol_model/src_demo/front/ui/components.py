import streamlit as st
from textwrap import dedent

from ..viz.charts import make_genre_donut, make_risk_donut


def render_header() -> None:
    st.markdown(
        dedent("""
            <div class="hero-wrap">
                <div class="hero-title">🎬 Netflix 고객 이탈 예측 서비스</div>
                <div class="hero-sub">고객 유지율, 위험군 분포, 행동 요인을 한 화면에서 확인합니다.</div>
                <div class="hero-insight">
                    현재 고위험 고객 비중은 12.0%이며, 최근 2주 이상 비활성 고객군의 위험도가 가장 높습니다.
                </div>
            </div>
        """),
        unsafe_allow_html=True,
    )


def render_kpi_card(title: str, value: str, delta: str, delta_type: str, icon: str) -> None:
    delta_class = "kpi-delta-pos" if delta_type == "positive" else "kpi-delta-neg"
    st.markdown(
        dedent(f"""
            <div class="kpi-card">
                <div class="kpi-top">
                    <div class="kpi-icon">{icon}</div>
                    <div class="{delta_class}">{delta}</div>
                </div>
                <div class="kpi-label">{title}</div>
                <div class="kpi-value">{value}</div>
            </div>
        """),
        unsafe_allow_html=True,
    )


def render_section_heading(title: str, subtitle: str) -> None:
    st.markdown(
        dedent(f"""
            <div class="section-heading-wrap">
                <div class="section-title">{title}</div>
                <div class="section-sub">{subtitle}</div>
            </div>
        """),
        unsafe_allow_html=True,
    )


# ── 카드 헤더만 HTML로 열기 / 닫기 ──────────────────────────────────────
# def _card_open() -> None:
#     st.markdown('<div class="card-box">', unsafe_allow_html=True)
#
# def _card_close() -> None:
#     st.markdown('</div>', unsafe_allow_html=True)
def card_container(key: str):
    return st.container(key=key)

def render_trend_card(title: str, subtitle: str, fig) -> None:
    """월별 이탈 추세 카드 — plotly chart 포함"""
    # _card_open()
    with st.container(key="card-trend"):
        render_section_heading(title, subtitle)
        st.plotly_chart(fig, use_container_width=True)
    # _card_close()


def render_risk_donut(risk_segments: dict) -> None:
    colors = ["#E50914", "#F97316", "#FACC15", "#22C55E"]
    total  = sum(risk_segments["values"])

    legend_html = ""
    for idx, (label, value) in enumerate(zip(risk_segments["labels"], risk_segments["values"])):
        pct = (value / total * 100) if total else 0
        legend_html += dedent(f"""
            <div class="legend-item">
                <div class="legend-left">
                    <div class="legend-dot" style="background:{colors[idx]};"></div>
                    <div class="legend-label">{label}</div>
                </div>
                <div class="legend-right">
                    <span class="legend-value">{value:,}</span>
                    <span class="legend-pct">{pct:.0f}%</span>
                </div>
            </div>
        """)

    # _card_open()
    # render_section_heading("이탈 위험 세분화", "위험도별 사용자 분포")
    # fig = make_risk_donut(risk_segments)
    # st.plotly_chart(fig, use_container_width=True)
    # st.markdown(legend_html, unsafe_allow_html=True)
    # _card_close()
    with st.container(key="card-risk-donut"):
        render_section_heading("이탈 위험 세분화", "위험도별 사용자 분포")
        fig = make_risk_donut(risk_segments)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(legend_html, unsafe_allow_html=True)


def render_ott_usage(ott_usage: list[dict]) -> None:
    rows_html = ""
    for item in ott_usage:
        rows_html += dedent(f"""
            <div class="metric-row">
                <div class="metric-row-top">
                    <div class="metric-label">{item["label"]}</div>
                    <div class="metric-value">{item["value"]}%</div>
                </div>
                <div class="bar-track">
                    <div class="bar-fill-blue" style="width:{item["value"]}%;"></div>
                </div>
            </div>
        """)

    # _card_open()
    # render_section_heading("OTT 이용 빈도", "최근 3개월 기준")
    # st.markdown(rows_html, unsafe_allow_html=True)
    # _card_close()
    with st.container(key="card-ott-usage"):
        render_section_heading("OTT 이용 빈도", "최근 3개월 기준")
        st.markdown(rows_html, unsafe_allow_html=True)


def render_genre_chart(genres: list[dict]) -> None:
    colors      = ["#E50914", "#8B5CF6", "#3B82F6", "#F59E0B", "#10B981"]
    legend_html = ""
    for idx, item in enumerate(genres):
        color = colors[idx % len(colors)]
        legend_html += dedent(f"""
            <div class="legend-item">
                <div class="legend-left">
                    <div class="legend-dot" style="background:{color};"></div>
                    <div class="legend-label">{item["label"]}</div>
                </div>
                <div class="legend-right">
                    <span class="legend-value">{item["value"]}%</span>
                </div>
            </div>
        """)

    # _card_open()
    # render_section_heading("콘텐츠 장르", "주요 시청 장르 비중")
    # fig = make_genre_donut(genres)
    # st.plotly_chart(fig, use_container_width=True)
    # st.markdown(legend_html, unsafe_allow_html=True)
    # _card_close()
    with st.container(key="card-genre-chart"):
        render_section_heading("콘텐츠 장르", "주요 시청 장르 비중")
        fig = make_genre_donut(genres)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(legend_html, unsafe_allow_html=True)


def render_churn_drivers(churn_drivers: list[dict]) -> None:
    rows_html = ""
    for item in churn_drivers:
        rows_html += dedent(f"""
            <div class="metric-row">
                <div class="metric-row-top">
                    <div class="metric-label">{item["label"]}</div>
                    <div class="metric-value">{item["value"]}%</div>
                </div>
                <div class="bar-track">
                    <div class="bar-fill-red" style="width:{item["value"]}%;"></div>
                </div>
            </div>
        """)

    # _card_open()
    # render_section_heading("이탈 예측 주요 요인", "이탈 가능성에 가장 큰 영향을 주는 행동 지표")
    # st.markdown(rows_html, unsafe_allow_html=True)
    # st.markdown(
    #     '<div class="alert-box">⚠ 최근 2주 이상 비활성 상태인 고객군의 이탈 위험이 가장 높습니다.</div>',
    #     unsafe_allow_html=True,
    # )
    # _card_close()
    with st.container(key="card-churn-drivers"):
        render_section_heading("이탈 예측 주요 요인", "이탈 가능성에 가장 큰 영향을 주는 행동 지표")
        st.markdown(rows_html, unsafe_allow_html=True)
        st.markdown(
            '<div class="alert-box">⚠ 최근 2주 이상 비활성 상태인 고객군의 이탈 위험이 가장 높습니다.</div>',
            unsafe_allow_html=True,
        )


def render_high_risk_users(users: list[dict]) -> None:
    render_section_heading("🚨 높은 이탈 위험 사용자", "위험도와 비활성 일수를 중심으로 핵심 고객 정보를 확인합니다.")

    for user in users:
        st.markdown(
            dedent(f"""
                <div class="user-card">
                    <div class="user-head">
                        <div class="user-head-left">
                            <div class="user-name">🔴 {user["name"]}</div>
                            <div class="user-subline">비활성 {user["inactive_days"]} · 마지막 활동 {user["last_active"]} · {user["genre"]}</div>
                        </div>
                        <div class="user-pill">{user["risk_label"]} · 위험도 {user["risk"]}</div>
                    </div>
                    <div class="user-grid">
                        <div class="user-meta highlight-meta">
                            <div class="user-meta-label">위험도</div>
                            <div class="user-meta-value">{user["risk"]}</div>
                        </div>
                        <div class="user-meta highlight-meta">
                            <div class="user-meta-label">비활성 일수</div>
                            <div class="user-meta-value">{user["inactive_days"]}</div>
                        </div>
                        <div class="user-meta">
                            <div class="user-meta-label">주요 장르</div>
                            <div class="user-meta-value">{user["genre"]}</div>
                        </div>
                        <div class="user-meta">
                            <div class="user-meta-label">이메일</div>
                            <div class="user-meta-value">{user["email"]}</div>
                        </div>
                    </div>
                </div>
            """).strip(),
            unsafe_allow_html=True,
        )

import streamlit as st

from ..state.state import go_dashboard, set_sample_mode, set_uploaded_file


def render_home_view() -> None:
    st.markdown(
        """
        <div class="landing-wrap">
            <div class="landing-badge">OTT Churn Prediction</div>
            <div class="landing-title">🎬 튈까말까 OTT 고객 이탈 예측 서비스</div>
            <div class="landing-sub">
                고객 데이터를 업로드하면 이탈 위험, 이용 패턴, 주요 이탈 요인을
                한눈에 확인할 수 있습니다.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, center, right = st.columns([1, 1.4, 1])

    with center:
        st.markdown(
            """
            <div class="upload-card">
                <div class="upload-card-title">데이터 업로드</div>
                <div class="upload-card-sub">
                    CSV 또는 XLSX 파일을 업로드한 뒤 분석 화면으로 이동합니다.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "고객 데이터 업로드",
            type=["csv", "xlsx"],
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            set_uploaded_file(uploaded_file)
            st.success(f"업로드 완료: {uploaded_file.name}")

        button_col1, button_col2 = st.columns(2)

        with button_col1:
            go_analysis = st.button(
                "분석 시작",
                type="primary",
                use_container_width=True,
                disabled=st.session_state.uploaded_file is None,
            )

        with button_col2:
            demo_mode = st.button(
                "샘플 데이터로 보기",
                use_container_width=True,
            )

        if go_analysis:
            go_dashboard()
            st.rerun()

        if demo_mode:
            set_sample_mode()
            go_dashboard()
            st.rerun()

import streamlit as st


def init_state() -> None:
    defaults = {
        # ── 화면 전환 ──────────────────────────────────────
        "current_view": "home",

        # ── 파일 업로드 ────────────────────────────────────
        "uploaded_file": None,
        "uploaded_file_name": None,
        "is_sample_mode": False,

        # ── 모델 선택 & 예측 결과 ──────────────────────────
        "selected_model": "XGBoost",        # "XGBoost" | "Random Forest" | "Logistic Regression"
        "model_results": None,              # 예측 완료된 DataFrame (churn_prob, risk_label 포함)
        "prediction_done": False,
        "prediction_error": None,

        # ── 대시보드 필터 ──────────────────────────────────
        "selected_risk": ["높은 위험", "중간 위험"],
        "selected_genres": ["드라마", "영화", "예능"],
        "selected_period": "최근 3개월",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def go_home() -> None:
    st.session_state.current_view = "home"


def go_dashboard() -> None:
    st.session_state.current_view = "dashboard"


def set_uploaded_file(uploaded_file) -> None:
    st.session_state.uploaded_file = uploaded_file
    st.session_state.uploaded_file_name = uploaded_file.name if uploaded_file else None
    st.session_state.is_sample_mode = False
    st.session_state.model_results = None
    st.session_state.prediction_done = False
    st.session_state.prediction_error = None


def set_sample_mode() -> None:
    st.session_state.uploaded_file = None
    st.session_state.uploaded_file_name = "sample_data"
    st.session_state.is_sample_mode = True
    st.session_state.model_results = None
    st.session_state.prediction_done = False
    st.session_state.prediction_error = None


def clear_uploaded_file() -> None:
    st.session_state.uploaded_file = None
    st.session_state.uploaded_file_name = None
    st.session_state.is_sample_mode = False
    st.session_state.model_results = None
    st.session_state.prediction_done = False
    st.session_state.prediction_error = None


def set_selected_model(model_name: str) -> None:
    if st.session_state.selected_model != model_name:
        st.session_state.selected_model = model_name
        st.session_state.model_results = None
        st.session_state.prediction_done = False
        st.session_state.prediction_error = None


def set_model_results(result_df) -> None:
    st.session_state.model_results = result_df
    st.session_state.prediction_done = True
    st.session_state.prediction_error = None


def set_prediction_error(msg: str) -> None:
    st.session_state.prediction_error = msg
    st.session_state.prediction_done = False


def reset_filters() -> None:
    st.session_state.selected_risk = ["높은 위험", "중간 위험"]
    st.session_state.selected_genres = ["드라마", "영화", "예능"]
    st.session_state.selected_period = "최근 3개월"


def reset_to_home() -> None:
    clear_uploaded_file()
    reset_filters()
    go_home()

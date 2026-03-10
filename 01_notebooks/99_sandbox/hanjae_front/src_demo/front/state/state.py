import streamlit as st


def init_state() -> None:
    defaults = {
        "current_view": "home",
        "uploaded_file": None,
        "uploaded_file_name": None,
        "is_sample_mode": False,
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
    st.session_state.uploaded_file_name = uploaded_file.name if uploaded_file is not None else None
    st.session_state.is_sample_mode = False


def set_sample_mode() -> None:
    st.session_state.uploaded_file = None
    st.session_state.uploaded_file_name = "sample_data"
    st.session_state.is_sample_mode = True


def clear_uploaded_file() -> None:
    st.session_state.uploaded_file = None
    st.session_state.uploaded_file_name = None
    st.session_state.is_sample_mode = False


def reset_filters() -> None:
    st.session_state.selected_risk = ["높은 위험", "중간 위험"]
    st.session_state.selected_genres = ["드라마", "영화", "예능"]
    st.session_state.selected_period = "최근 3개월"


def reset_to_home() -> None:
    clear_uploaded_file()
    reset_filters()
    go_home()
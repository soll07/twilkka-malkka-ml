import streamlit as st

from src_demo.front.state.state import init_state
from src_demo.front.views.dashboard_view import render_dashboard_view
from src_demo.front.views.home_view import render_home_view
from src_demo.front.viz.styles import inject_css


def main() -> None:
    st.set_page_config(
        page_title="Netflix 고객 이탈 예측 서비스",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    inject_css()
    init_state()

    if st.session_state.current_view == "home":
        render_home_view()
    else:
        render_dashboard_view()


if __name__ == "__main__":
    main()
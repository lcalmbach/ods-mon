import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import json
from pathlib import Path

from dataset import Dataset
from texts import txt

__version__ = "0.0.1"
__author__ = "Lukas Calmbach"
__author_email__ = "lcalmbach@gmail.com"

VERSION_DATE = "2024-09-09"
APP_NAME = "ODS-Mon"
APP_ICON = "📈"
GIT_REPO = "https://github.com/lcalmbach/ods-mon"

menu_icons = ["house", "gear", "chat-dots"]
menu_options = [
    "About",
    "Report Settings",
    "Generate Reports",
]
settings_file = Path("monitor.json")
settings_dict = json.loads(settings_file.read_text())

APP_INFO = f"""<div style="background-color:#34282C; padding: 10px;border-radius: 15px; border:solid 1px white;">
    <small>Created by <a href="mailto:{__author_email__}">{__author__}</a><br>
    Version: {__version__} ({VERSION_DATE})<br>
    Source data: <a href="https://data.bs.ch">data.bs</a><br>
    <a href="{GIT_REPO}">git-repo</a></small></div>
    """


def init():
    st.set_page_config(
        page_title=APP_NAME,
        page_icon=APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )


def main():
    init()
    with st.sidebar:
        menu_action = option_menu(
            None,
            menu_options,
            icons=menu_icons,
            menu_icon="cast",
            default_index=0,
        )
    index = menu_options.index(menu_action)
    if index == 0:
        col1, col2, col3 = st.columns([1, 4, 1])
        image_path = "./ods-mon-splashscreen.webp"
        with col2:
            st.image(image_path, width=600)
        st.markdown(txt["info"], unsafe_allow_html=True)
    elif index == 1:
        st.subheader("Reports Configuration")
        text = json.loads(settings_file.read_text())
        st.write(text)
    elif index == 2:
        dataset = st.selectbox(
            "Select a ODS-dataset",
            settings_dict.keys(),
            format_func=lambda x: settings_dict[x]["title"],
        )

        if "dataset" not in st.session_state:
            st.session_state.dataset = Dataset(dataset, settings_dict[dataset])
        if st.session_state.dataset.key != dataset:
            st.session_state.dataset = Dataset(dataset, settings_dict[dataset])
        date = st.sidebar.date_input(
            "Select a date", value=st.session_state.dataset.get_last_date()
        )
        if date != st.session_state.dataset.date:
            st.session_state.dataset.date = date
        ds = st.session_state.dataset
        num_days_preview = st.sidebar.slider("Number of days to preview", 1, 100, 7)
        with st.expander(f"description of the dataset", expanded=False):
            st.markdown(
                ds.ods_metadata["metas"]["default"]["description"],
                unsafe_allow_html=True,
            )
            st.markdown(f"[Dataset on ODS portal]({ds.home_url()})")
        with st.expander(f"Preview last {num_days_preview} days", expanded=True):
            df = ds.get_last_n_records(num_days_preview)
            if len(df) > 0:
                st.dataframe(df, hide_index=True)
            else:
                st.markdown("No data available for the selected date.")
        if st.sidebar.button("Generate Report", key="generate_report"):
            report_results = ds.get_reports()
            for report in report_results:
                st.write(report[0])
                st.dataframe(report[1], hide_index=True)
                st.write(report[2])
    st.sidebar.markdown(APP_INFO, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

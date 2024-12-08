import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import json
from pathlib import Path
import random
import string
import os

from dataset import Dataset, ReportType, MONTH2SEASON
from texts import txt
import gpt

from helper import MONTHS_DICT, SEASON_DICT

__version__ = "0.1.0"
__author__ = "Lukas Calmbach"
__author_email__ = "lcalmbach@gmail.com"

VERSION_DATE = "2024-12-058"
APP_NAME = "Data-News-Generator"
APP_ICON = "📰"
GIT_REPO = "https://github.com/lcalmbach/ods-mon"

menu_icons = ["house", "gear", "chat-dots"]
menu_options = [
    "About",
    "Report Settings",
    "Generate Reports",
]
settings_file = Path("monitor.json")
settings_dict = json.loads(settings_file.read_text())
AUDIO_PATH = "./assets/audio/"
CSS_FILE = "./assets/css/style.css"
MODEL = "GPT-4o"
SPLASH_IMAGE = "./assets/images/splashscreen.webp"

APP_INFO = f"""<div style="background-color:#CFEFF4; padding: 10px;border-radius: 15px; border:solid 1px white;">
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
    with open(CSS_FILE) as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
    # make sure lastmenu is initialized so it can be compared
    if "last_action" not in st.session_state:
        st.session_state.last_action = None


def get_year(ds):
    years_options = ds.get_year_range()
    default_year = ds.date.year
    index = years_options.index(default_year)
    return st.sidebar.selectbox("Select a Year", options=years_options, index=index)


def get_month(ds):
    default_month = ds.date.month
    index = range(1, 13).index(default_month)
    return st.sidebar.selectbox(
        "Select a Month",
        options=MONTHS_DICT.keys(),
        format_func=lambda x: MONTHS_DICT[x],
        index=index,
    )


def get_season(ds):
    default_season = MONTH2SEASON[ds.date.month]
    index = range(1, 5).index(default_season)
    return st.sidebar.selectbox(
        "Select a Season",
        options=SEASON_DICT.keys(),
        format_func=lambda x: SEASON_DICT[x],
        index=index,
    )


def get_audio_filename()->str:
    """
    Generates a random filename for the audio file in folder ./assets/audio.
    Returns:
        str: The filename for the audio file."""
    return (
        AUDIO_PATH
        + "".join(random.choices(string.ascii_letters + string.digits, k=8))
        + ".wav"
    )


def show_report():
    dataset = st.selectbox(
        "Select an ODS-Dataset",
        settings_dict.keys(),
        format_func=lambda x: settings_dict[x]["title"],
    )

    if "dataset" not in st.session_state:
        st.session_state.dataset = Dataset(dataset, settings_dict[dataset])
    if st.session_state.dataset.key != dataset:
        st.session_state.dataset = Dataset(dataset, settings_dict[dataset])
        if "result" in st.session_state:
            del st.session_state.result

    ds = st.session_state.dataset
    report = st.sidebar.radio("Select a report", ds.get_report_types())
    if report == ReportType.DAILY.value:
        date = st.sidebar.date_input(
            "Select a Date", value=st.session_state.dataset.get_last_date()
        )
        if date != ds.date:
            ds.date = date
    elif report == ReportType.MONTHLY.value:
        month = get_month(ds)
        if month != ds.month:
            ds.month = month
        year = get_year(ds)
        if year != ds.year:
            ds.year = year
    elif report == ReportType.SEASON.value:
        season = get_season(ds)
        if season != ds.season:
            ds.season = season
        year = get_year(ds)
        if year != ds.year:
            ds.year = year
    elif report == ReportType.YEARLY.value:
        year = get_year(ds)
        if year != ds.year:
            ds.year = year

    generate_summary = st.sidebar.checkbox("Generate AI summary", value=True)
    if generate_summary:
        generate_audio = st.sidebar.checkbox("Text2Speech", value=True)
    with st.expander(f"Description and preview of the Dataset", expanded=False):
        st.markdown(
            ds.ods_metadata["metas"]["default"]["description"],
            unsafe_allow_html=True,
        )
        st.markdown(f"Number of Records: {ds.get_num_records()}")
        min_date, max_date = ds.get_date_range()
        st.markdown(
            f"Date of first and last record: {min_date.strftime('%Y-%m-%d')} - {max_date.strftime('%Y-%m-%d')}"
        )
        st.markdown(f"[Dataset on ODS portal]({ds.home_url()})")
        num_days_preview = st.slider("Number of days to preview", 1, 100, 7)
        st.markdown(f"---\n**Preview of last {num_days_preview} records**")
        df = ds.get_last_n_records(num_days_preview)
        if len(df) > 0:
            st.dataframe(df, hide_index=True)
        else:
            st.warning("No data available for the selected date.")

    if st.button("Generate Reports", key="generate_report"):
        with st.spinner("Generating Reports..."):
            st.session_state.result = ds.get_reports(report, generate_summary)
    if "result" in st.session_state:
        result = st.session_state.result
        st.markdown(result["title"])
        st.dataframe(result["data"], hide_index=True)
        st.markdown(txt["table_legend"], unsafe_allow_html=True)

        st.download_button(
            label="Download CSV",
            data=result["data"].to_csv(index=False),
            file_name=f"{ds.key}_{report}.csv",
            mime="text/csv",
        )
        if generate_summary:
            st.markdown("---")
            st.markdown(f"### Report Summary (AI-generated Text using {MODEL}):")
            st.markdown(result["summary_text"])
            if generate_audio:
                if not "audio_file" in st.session_state:
                    st.session_state.audio_file = get_audio_filename()
                    _ = gpt.get_audio(
                        result["summary_text"], st.session_state.audio_file
                    )
                if "audio_file" in st.session_state and os.path.exists(st.session_state.audio_file):
                    file_path = Path(st.session_state.audio_file)
                    if file_path.exists():
                        st.audio(st.session_state.audio_file, format="audio/wav")
                        st.download_button(
                            label="Download Audio",
                            data=st.session_state.audio_file,
                            file_name="news.wav",
                            mime="audio/wav",
                        )


def main():
    init()
    with st.sidebar:
        st.session_state.menu_action = option_menu(
            None,
            menu_options,
            icons=menu_icons,
            menu_icon=APP_ICON,
            default_index=0,
        )
    index = menu_options.index(st.session_state.menu_action)
    # remove current result if a different option is selected.
    if (
        "result" in st.session_state
        and st.session_state.menu_action != st.session_state.last_action
    ):
        del st.session_state.result
    if index == 0:
        col1, col2, col3 = st.columns([1, 4, 1])
        image_path = SPLASH_IMAGE
        with col2:
            st.image(image_path, width=600)
        st.markdown(txt["info"], unsafe_allow_html=True)
    elif index == 1:
        st.subheader("Reports Configuration")
        st.markdown(txt["settings_info"])
        text = json.loads(settings_file.read_text())
        st.write(text)
    elif index == 2:
        show_report()
    st.session_state.last_action = st.session_state.menu_action
    st.sidebar.markdown(APP_INFO, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

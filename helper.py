import streamlit as st
import socket
import os

LOCAL_HOST = "desktop-fo7m3a8"

MONTHS_DICT = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec",
}
SEASON_DICT = {1: "Winter", 2: "Spring", 3: "Summer", 4: "Fall"}


def get_var(varname: str) -> str:
    """
    Retrieves the value of a given environment variable or secret from the Streamlit configuration.

    If the current host is the local machine (according to the hostname), the environment variable is looked up in the system's environment variables.
    Otherwise, the secret value is fetched from Streamlit's secrets dictionary.

    Args:
        varname (str): The name of the environment variable or secret to retrieve.

    Returns:
        The value of the environment variable or secret, as a string.

    Raises:
        KeyError: If the environment variable or secret is not defined.
    """
    if socket.gethostname().lower() == LOCAL_HOST:
        return os.environ[varname]
    else:
        return st.secrets[varname]

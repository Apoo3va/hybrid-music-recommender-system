import time
import requests

APP_URL = "http://localhost:8000"


def get_status_code(url):
    response = requests.get(url)
    return response.status_code


def test_streamlit_app_loads():
    time.sleep(60)
    status_code = get_status_code(APP_URL)
    assert status_code == 200, "Unable to load the Streamlit app."
    print("Streamlit app loaded successfully.")
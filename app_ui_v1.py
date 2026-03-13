import streamlit as st
import requests
import time
import json
import pandas as pd

# --- Configuration ---
BACKEND_URL = "http://127.0.0.1:8000"
API_COMMAND_URL = f"{BACKEND_URL}/api/command/"
API_STATUS_URL = f"{BACKEND_URL}/api/get_info/"
API_LOG_URL = f"{BACKEND_URL}/api/get_latest_log/"

# --- Helper Functions ---
def send_command(command, args="{}"):
    """Sends a command to the backend."""
    try:
        response = requests.post(API_COMMAND_URL, json={"command": command, "args": args})
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("Connection to backend failed. Is the trainer running?")
        return None

def get_status():
    """Gets the current status from the backend."""
    try:
        response = requests.get(API_STATUS_URL)
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"status": "disconnected"}

def get_latest_log():
    """Gets the latest log metrics from the backend."""
    try:
        response = requests.get(API_LOG_URL)
        if response.text:
            return response.json()
        return {}
    except (requests.exceptions.ConnectionError, json.JSONDecodeError):
        return {}

# --- Initialize Session State ---
if 'log_history' not in st.session_state:
    st.session_state.log_history = []
if 'last_epoch' not in st.session_state:
    st.session_state.last_epoch = -1
if 'w_accuracy' not in st.session_state:
    st.session_state.w_accuracy = 1.0
if 'w_fairness' not in st.session_state:
    st.session_state.w_fairness = 0.5

# --- UI Layout ---
st.set_page_config(layout="wide")

st.title("⚖️ Interactive AI Training Dashboard")
st.caption("Dynamically trade-off Accuracy, Fairness, and Energy.")

# --- Main Content Area ---
status_placeholder = st.empty()
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📊 Live Metrics")
    metrics_placeholder = st.empty()
    chart_placeholder = st.empty()

with col2:
    st.header("⚙️ Controls")

    # --- Control Buttons ---
    btn_cols = st.columns(3)
    with btn_cols[0]:
        if st.button("Start Training", use_container_width=True):
            send_command("start_training")
        if st.button("Pause Training", use_container_width=True):
            send_command("pause_training")
    with btn_cols[1]:
        if st.button("Resume Training", use_container_width=True):
            send_command("resume_training")
    with btn_cols[2]:
        if st.button("🛑 Stop Training", type="primary", use_container_width=True):
            send_command("stop_training")
            st.session_state.log_history = [] # Clear logs on stop

    st.divider()

    # --- Weight Sliders ---
    st.subheader("Objective Weights")
    
    def update_weights():
        """Callback to send weight updates."""
        weights = {
            "accuracy": st.session_state.w_accuracy,
            "fairness": st.session_state.w_fairness
        }
        send_command("update_weights", json.dumps(weights))

    st.slider(
        "Accuracy Weight", 0.0, 10.0,
        key='w_accuracy',
        on_change=update_weights
    )
    st.slider(
        "Fairness Weight", 0.0, 10.0,
        key='w_fairness',
        on_change=update_weights
    )


# --- Main Application Loop ---
while True:
    status_data = get_status()
    log_data = get_latest_log()

    # Update Status Badge
    status = status_data.get('status', 'disconnected').capitalize()
    if status == 'Running':
        status_placeholder.success(f"**Status:** {status}", icon="🏃")
    elif status == 'Paused':
        status_placeholder.warning(f"**Status:** {status}", icon="⏸️")
    elif status == 'Finished':
        status_placeholder.info(f"**Status:** {status}", icon="🏁")
    else:
        status_placeholder.error(f"**Status:** {status}", icon="🔌")

    # Update metrics and chart if new log data is available
    if log_data and log_data.get('epoch', -1) > st.session_state.last_epoch:
        st.session_state.last_epoch = log_data['epoch']
        st.session_state.log_history.append(log_data)

        with metrics_placeholder.container():
            m_cols = st.columns(4)
            m_cols[0].metric("Epoch", f"{log_data.get('epoch', 0)}")
            m_cols[1].metric("Accuracy", f"{log_data.get('accuracy', 0):.3f}")
            m_cols[2].metric("Fairness (DPD)", f"{log_data.get('demographic_parity_difference', 0):.3f}")
            m_cols[3].metric("Energy (mJ)", f"{log_data.get('energy_mJ', 0):.2f}")

    # Draw chart from history
    if st.session_state.log_history:
        df = pd.DataFrame(st.session_state.log_history).set_index('epoch')
        chart_placeholder.line_chart(df[['accuracy', 'demographic_parity_difference']])

    if status == 'Finished' or status == 'Disconnected':
        st.stop()

    time.sleep(1) # Poll every 1 second
import streamlit as st
import requests
import time
import pandas as pd
import json

# --- Config & Theming ---
st.set_page_config(layout="wide", page_title="MOO v4 | Interactive Dashboard")

# Comprehensive minimalist techy CSS
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .metric-card {
        background-color: #161b22;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #30363d;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #161b22;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #21262d;
        border-bottom: 2px solid #58a6ff;
    }
    </style>
""", unsafe_allow_html=True)

BACKEND_URL = "http://127.0.0.1:5000"

# --- Session State ---
if 'history' not in st.session_state:
    st.session_state.history = {}

def fetch_models():
    try: return requests.get(f"{BACKEND_URL}/api/models").json()
    except: return {}

def send_cmd(mid, cmd, args="{}"):
    requests.post(f"{BACKEND_URL}/api/command/", json={"model_id": mid, "command": cmd, "args": args})

def delete_model(mid):
    requests.delete(f"{BACKEND_URL}/api/delete/{mid}")
    if mid in st.session_state.history:
        del st.session_state.history[mid]

# --- Sidebar Controller ---
with st.sidebar:
    st.title("Controller")
    st.divider()
    st.subheader("Add New Model")
    new_id = st.text_input("Instance Name", value=f"Node_{len(st.session_state.history) + 1}")
    new_epochs = st.number_input("Iter Cycles", min_value=10, max_value=1000, value=200)
    
    if st.button("Add Model", use_container_width=True):
        resp = requests.post(f"{BACKEND_URL}/api/create/{new_id}/{new_epochs}").json()
        if resp.get("status") == "started":
            st.session_state.history[new_id] = []
            st.success(f"Initialized {new_id}")
        else:
            st.error("Model ID already exists")
    
    st.divider()
    
    # Global Delete Dropdown
    active_models = fetch_models()
    if active_models:
        st.subheader("Manage Instances")
        delete_id = st.selectbox("Select model to remove", options=list(active_models.keys()))
        if st.button("Delete Instance", use_container_width=True, type="secondary"):
            delete_model(delete_id)
            st.toast(f"Model {delete_id} removed")
            st.rerun()

    st.divider()
    st.info("System Ready. High-fidelity tracking active.")

# --- Main Dashboard ---
st.title("Multi-Objective Optimization Engine")
st.caption("Fidelity-driven balancing of Accuracy, Fairness, Bias, and Energy Consumption.")

if not active_models:
    st.empty()
    st.markdown("""
        <div style='padding: 50px; text-align: center; color: #8b949e;'>
            <h2>SYSTEM STANDBY</h2>
            <p>Deploy a neural instance from the controller to initiate telemetry.</p>
        </div>
    """, unsafe_allow_html=True)
else:
    model_ids = list(active_models.keys())
    tabs = st.tabs([f"Instance: {mid}" for mid in model_ids])

    for i, mid in enumerate(model_ids):
        with tabs[i]:
            status_info = active_models[mid]
            log = requests.get(f"{BACKEND_URL}/api/logs/{mid}").json()
            
            # History Buffer update
            if log and (not st.session_state.history.get(mid) or log['epoch'] != st.session_state.history[mid][-1]['epoch']):
                if mid not in st.session_state.history: st.session_state.history[mid] = []
                st.session_state.history[mid].append(log)

            # Metrics Overview
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            with m_col1:
                st.metric("Status", status_info['status'].upper())
            with m_col2:
                prog = log.get('epoch', 0)
                tot = log.get('total_epochs', status_info.get('total', 100))
                st.metric("Progress", f"{prog}/{tot}")
            with m_col3:
                acc = log.get('accuracy', 0)
                st.metric("Accuracy", f"{acc:.2%}")
            with m_col4:
                nrg = log.get('energy_consumed', 0)
                st.metric("Energy used", f"{nrg:.6f} kWh")

            st.divider()

            # Layout: Analytics and Control
            col_graphs, col_tuning = st.columns([3, 1])

            with col_graphs:
                if st.session_state.history[mid]:
                    df = pd.DataFrame(st.session_state.history[mid])
                    
                    # Chart Row 1
                    g1, g2 = st.columns(2)
                    with g1:
                        st.write("Model Loss")
                        st.line_chart(df.set_index('epoch')['loss'], color="#58a6ff")
                    with g2:
                        st.write("Fairness (DPD)")
                        st.line_chart(df.set_index('epoch')['fairness'], color="#3fb950")
                    
                    # Chart Row 2
                    g3, g4 = st.columns(2)
                    with g3:
                        st.write("Bias (Sex Mean Diff)")
                        st.line_chart(df.set_index('epoch')['bias'], color="#f85149")
                    with g4:
                        st.write("Energy Trend")
                        st.line_chart(df.set_index('epoch')['energy_consumed'], color="#d29922")
                else:
                    st.info("Awaiting telemetry stream. Start training to see live analytics.")

            with col_tuning:
                with st.container(border=True):
                    st.write("Execution")
                    if st.button("Start Training", key=f"start_{mid}", use_container_width=True):
                        send_cmd(mid, "start_training")
                    
                    c1, c2 = st.columns(2)
                    if c1.button("Pause", key=f"p_{mid}", use_container_width=True): send_cmd(mid, "pause_training")
                    if c2.button("Resume", key=f"r_{mid}", use_container_width=True): send_cmd(mid, "resume_training")
                    
                    if st.button("Stop Training", key=f"s_{mid}", use_container_width=True): 
                        send_cmd(mid, "stop_training")

                st.write("")
                with st.container(border=True):
                    st.write("Tuning")
                    acc_w = st.slider("Accuracy Weight", 0.0, 10.0, 1.0, key=f"aw_{mid}")
                    fair_w = st.slider("Fairness Weight", 0.0, 10.0, 0.5, key=f"fw_{mid}")
                    nrgy_w = st.slider("Energy Weight", 0.0, 10.0, 0.5, key=f"ew_{mid}")
                    
                    if st.button("Update Weights", key=f"btn_{mid}", use_container_width=True):
                        send_cmd(mid, "update_weights", json.dumps({"accuracy": acc_w, "fairness": fair_w, "energy": nrgy_w}))
                        st.toast("Weights updated")

                st.write("")
                if st.button("Delete Model", key=f"del_{mid}", use_container_width=True, type="secondary"):
                    delete_model(mid)
                    st.rerun()

# Refresher
time.sleep(1)
st.rerun()

import streamlit as st
import requests
import time
import pandas as pd
import json

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(layout="wide", page_title="Interactive AI Multi-Trainer")

# --- Session State ---
if 'history' not in st.session_state:
    st.session_state.history = {} # model_id -> list of logs

def fetch_models():
    try: return requests.get(f"{BACKEND_URL}/api/models").json()
    except: return {}

def send_cmd(mid, cmd, args="{}"):
    requests.post(f"{BACKEND_URL}/api/command/", json={"model_id": mid, "command": cmd, "args": args})

# --- Sidebar ---
with st.sidebar:
    st.header("🚢 Model Launcher")
    new_id = st.text_input("Model Name", value=f"Model_{len(st.session_state.history) + 1}")
    new_epochs = st.number_input("Total Epochs", min_value=10, max_value=1000, value=100)
    
    if st.button("🚀 Start Training Session"):
        resp = requests.post(f"{BACKEND_URL}/api/create/{new_id}/{new_epochs}").json()
        if resp.get("status") == "started":
            st.session_state.history[new_id] = []
            st.success(f"Launched {new_id}")
        else:
            st.error("Failed to launch (check if ID exists)")

st.title("⚖️ Multi-Objective Optimization Dashboard")

active_models = fetch_models()

if not active_models:
    st.info("No active models. Create one in the sidebar to begin.")
else:
    # Creating Dynamic Tabs
    model_ids = list(active_models.keys())
    tabs = st.tabs(model_ids)

    for i, mid in enumerate(model_ids):
        with tabs[i]:
            # Data Fetching
            status_info = active_models[mid]
            log = requests.get(f"{BACKEND_URL}/api/logs/{mid}").json()
            
            # Store history for charting
            if log and (not st.session_state.history.get(mid) or log['epoch'] != st.session_state.history[mid][-1]['epoch']):
                if mid not in st.session_state.history: st.session_state.history[mid] = []
                st.session_state.history[mid].append(log)

            # UI Layout
            col_metrics, col_ctrl = st.columns([3, 1])
            
            with col_metrics:
                st.subheader(f"Live Performance: {mid}")
                if log:
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Status", status_info['status'].upper())
                    m2.metric("Progress", f"{log['epoch']}/{log['total_epochs']}")
                    m3.metric("Accuracy", f"{log['accuracy']:.3%}")
                    m4.metric("Fairness (DPD)", f"{log['fairness']:.4f}")
                    
                    e1, e2 = st.columns(2)
                    e1.metric("Energy (kWh)", f"{log['energy_consumed']:.6f}")
                    e2.metric("Power (W)", f"{log['power_draw']:.2f}")
                    
                    # Graphing
                    df = pd.DataFrame(st.session_state.history[mid])
                    st.line_chart(df.set_index('epoch')[['accuracy', 'fairness', 'energy_consumed']])
                else:
                    st.warning("Waiting for first epoch...")

            with col_ctrl:
                st.subheader("Interactive Tuning")
                
                # Manual Controls
                c1, c2 = st.columns(2)
                if c1.button("Pause", key=f"p_{mid}"): send_cmd(mid, "pause_training")
                if c2.button("Resume", key=f"r_{mid}"): send_cmd(mid, "resume_training")
                if st.button("🛑 Stop Training", key=f"s_{mid}", use_container_width=True): send_cmd(mid, "stop_training")
                
                st.divider()
                
                # Weight Sliders
                st.write("**Adjust Objective Weights**")
                acc_w = st.slider("Accuracy Weight", 0.0, 10.0, 1.0, key=f"aw_{mid}")
                fair_w = st.slider("Fairness Weight", 0.0, 10.0, 0.5, key=f"fw_{mid}")
                nrgy_w = st.slider("Energy Weight", 0.0, 10.0, 0.5, key=f"ew_{mid}")
                
                if st.button("Update Weights", key=f"btn_{mid}"):
                    send_cmd(mid, "update_weights", json.dumps({"accuracy": acc_w, "fairness": fair_w, "energy": nrgy_w}))
                    st.toast(f"Weights updated for {mid}!")

# Auto-Refresh
time.sleep(1)
st.rerun()
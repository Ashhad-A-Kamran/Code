import streamlit as st
import requests
import time
import pandas as pd
import json

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(layout="wide", page_title="Multi-Model MOO")

# --- Session State Management ---
if 'models' not in st.session_state:
    st.session_state.models = {} # model_id -> log_history list

def get_all_models():
    try: return requests.get(f"{BACKEND_URL}/api/models").json()
    except: return {}

def send_command(mid, cmd, args="{}"):
    requests.post(f"{BACKEND_URL}/api/command/", json={"model_id": mid, "command": cmd, "args": args})

# --- Sidebar: Model Management ---
with st.sidebar:
    st.header("Model Management")
    new_model_name = st.text_input("New Model ID", value="Model_1")
    if st.button("🚀 Launch New Model"):
        requests.post(f"{BACKEND_URL}/api/create/{new_model_name}")
        st.session_state.models[new_model_name] = []
    
    st.divider()
    active_models = get_all_models()
    st.write(f"Active Models: {len(active_models)}")

st.title("Interactive Multi-Model Dashboard")

if not active_models:
    st.info("No models running. Use the sidebar to start one.")
else:
    # Create tabs for each model
    tabs = st.tabs(list(active_models.keys()))

    for i, model_id in enumerate(active_models.keys()):
        with tabs[i]:
            m_status = active_models[model_id]['status']
            col1, col2 = st.columns([2, 1])
            
            # Fetch latest logs
            log = requests.get(f"{BACKEND_URL}/api/logs/{model_id}").json()
            if log and (not st.session_state.models.get(model_id) or log['epoch'] != st.session_state.models[model_id][-1]['epoch']):
                if model_id not in st.session_state.models: st.session_state.models[model_id] = []
                st.session_state.models[model_id].append(log)

            with col1:
                st.subheader(f"Metrics: {model_id} ({m_status})")
                if log:
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Accuracy", f"{log.get('accuracy', 0):.3f}")
                    m2.metric("Fairness (DPD)", f"{log.get('fairness', 0):.3f}")
                    m3.metric("Epoch", log.get('epoch', 0))
                
                if st.session_state.models[model_id]:
                    df = pd.DataFrame(st.session_state.models[model_id])
                    st.line_chart(df.set_index('epoch')[['accuracy', 'fairness']])

            with col2:
                st.subheader("Controls")
                c1, c2 = st.columns(2)
                if c1.button("Pause", key=f"p_{model_id}"): send_command(model_id, "pause_training")
                if c2.button("Resume", key=f"r_{model_id}"): send_command(model_id, "resume_training")
                if st.button("🛑 Stop", key=f"s_{model_id}", type="primary"): send_command(model_id, "stop_training")
                
                st.divider()
                st.write("Weight Tuning")
                
                # Use a local function to handle weight changes
                acc_w = st.slider("Accuracy Weight", 0.0, 5.0, 1.0, key=f"acc_{model_id}")
                fair_w = st.slider("Fairness Weight", 0.0, 5.0, 0.5, key=f"fair_{model_id}")
                
                if st.button("Apply Weights", key=f"upd_{model_id}"):
                    send_command(model_id, "update_weights", json.dumps({"accuracy": acc_w, "fairness": fair_w}))

# --- Auto-refresh logic ---
time.sleep(1)
st.rerun()
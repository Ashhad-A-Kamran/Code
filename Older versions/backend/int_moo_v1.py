import os
import uuid
import time
import json
import queue
import asyncio
import threading
import numpy as np
import pandas as pd
from typing import Dict, List


# --- Machine Learning and Metrics Imports ---
import torch
import torch.nn as nn
import torch.optim as optim
from fairlearn.datasets import fetch_adult
from fairlearn.metrics import demographic_parity_difference
# from pyJoules.energy_meter import EnergyMeter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# from pyJoules.device.rapl_device import RaplPackageDomain

# devices_to_monitor = [RaplPackageDomain(0)]

# --- Interactive Server Imports ---
from uvicorn import Config, Server
from dataclasses import dataclass, field
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, APIRouter

from pydantic import BaseModel


CMD_SUCCESS = "success"
COMMAND_TO_TYPE = {
    "start_training": "wrapper_control_command_type",
    "update_weights": "update_weights_type",
    "pause_training": "pause_resume_type",
    "resume_training": "pause_resume_type",
    "stop_training": "wrapper_control_command_type"
}
UPDATE_WEIGHTS_TYPE = "update_weights_type"
PAUSE_RESUME_TYPE = "pause_resume_type"
WRAPER_CONTROL_COMMAND_TYPE = "wrapper_control_command_type"
PAUSE_TRAINING = "pause_training"
RESUME_TRAINING = "resume_training"
STOP_TRAINING = "stop_training"


import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Cmd:
    command: str
    args: str = "{}"
    uuid: str = field(default_factory=lambda: str(uuid.uuid4()))
    time: float = field(default_factory=time.time)
    status: str = "ok"


class CmdModel(BaseModel):
    command: str
    args: str = "{}"
    uuid: str = None
    time: float = None
    status: str = "ok"

    class Config:
        arbitrary_types_allowed = True

# --- Interactive Server Code ---

@dataclass
class InteractiveServerState:
    """
    Stores the current training state
    """

    status: str = "init"
    run_name: str = "Interactive MOO Training"
    weights: Dict[str, float] = field(default_factory=lambda: {"accuracy": 1.0, "fairness": 0.5})
    latest_log: dict = field(default_factory=dict) # <<< Store latest log


class InteractiveServer:
    """
    Main Responsibilities
        Start FastAPI server
        Provide REST endpoints
        Provide WebSocket streaming
        Manage event queues
        Manage training state
        Route commands to trainer
    """

    def __init__(self, host: str, port: int):
        self.app = FastAPI()
        self.app.add_middleware(
            CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
        )
        self.host = host
        self.port = port
        self.events_queue: "queue.Queue[dict]" = queue.Queue()
        self.messages_queue_by_type: Dict[str, "queue.Queue[Cmd]"] = {
            UPDATE_WEIGHTS_TYPE: queue.Queue(),
            PAUSE_RESUME_TYPE: queue.Queue(),
            WRAPER_CONTROL_COMMAND_TYPE: queue.Queue(),
        }
        self._train_state = InteractiveServerState()
        self._train_state_lock = threading.Lock()
        self._event_listeners: set[WebSocket] = set()
        self.server: Server | None = None
        self.running = False
        self._setup_routes()
    
    # Method to update the latest log
    def set_latest_log(self, log_event: dict):
        with self._train_state_lock:
            self._train_state.latest_log = log_event

    def enqueue_event(self, event: dict):
        self.events_queue.put(event)

    def update_server_state(self, event: dict):
        with self._train_state_lock:
            if event["command"] == PAUSE_TRAINING:
                self._train_state.status = "paused"
            elif event["command"] == RESUME_TRAINING:
                self._train_state.status = "running"
            elif event["command"] == "update_weights":
                 new_weights = json.loads(event.get("args", "{}"))
                 self._train_state.weights.update(new_weights)

    def _start_server(self):
        config = Config(app=self.app, host=self.host, port=self.port, loop="asyncio")
        self.server = Server(config=config)
        self.server.run()

    def _setup_routes(self):
        api_router = APIRouter(prefix="/api")
        ws_router = APIRouter(prefix="/ws")

        @api_router.get("/get_info/")
        async def get_train_state():
            with self._train_state_lock:
                return {"status": self._train_state.status, "run_name": self._train_state.run_name}
        
        @api_router.get("/get_weights/")
        async def get_weights():
             with self._train_state_lock:
                  return self._train_state.weights
        
        # Endpoint to get the latest log for polling
        @api_router.get("/get_latest_log/")
        async def get_latest_log():
            with self._train_state_lock:
                return self._train_state.latest_log

        @api_router.post("/command/")
        async def receive_command(cmd: CmdModel):
            logger.info(f"Received command over HTTP: {cmd.model_dump_json()}")
# Websocket
        @ws_router.websocket("/ws/message/")
        async def websocket_message(websocket: WebSocket):
            await websocket.accept()
            self._event_listeners.add(websocket)
            loop = asyncio.get_running_loop()
            try:
                while True:
                    event = await loop.run_in_executor(None, self.events_queue.get)
                    if event is None: break
                    await asyncio.gather(*(ws.send_text(json.dumps(event)) for ws in self._event_listeners))
            finally:
                self._event_listeners.discard(websocket)

        self.app.include_router(api_router)
        self.app.include_router(ws_router)

    def run(self):
        if not self.running:
            self.running = True
            thread = threading.Thread(target=self._start_server, daemon=True)
            thread.start()

    def stop(self):
        if self.running and self.server:
            self.running = False
            self.server.should_exit = True
            self.events_queue.put(None)


class LogisticRegression(nn.Module):
    def __init__(self, n_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


class InteractiveTrainer:
    def __init__(self, server: InteractiveServer):
        self.server = server
        self.model = None
        self.optimizer = None
        self.criterion = nn.BCELoss()

        # Handle start command
        try:
            cmd = self.server.messages_queue_by_type[WRAPER_CONTROL_COMMAND_TYPE].get_nowait()
            if cmd.command == "start_training":
                if not hasattr(self, "training_thread") or not self.training_thread.is_alive():
                    self.training_thread = threading.Thread(target=self.train, daemon=True)
                    self.training_thread.start()
                    logger.info("Training started from command.")
        except queue.Empty:
            pass

        # Training control flags
        self.is_paused = False
        self.should_stop = False

        # Interactive weights
        self.w_accuracy = 1.0
        self.w_fairness = 0.5

        # Data placeholders
        self.X_train_tensor, self.y_train_tensor, self.X_test_tensor, self.y_test = (None,) * 4
        self.sex_train, self.sex_test = None, None
        self.privileged_mask_train, self.unprivileged_mask_train = None, None

    def _load_and_preprocess_data(self):
        logger.info("Loading and preprocessing Adult Census dataset...")
        data = fetch_adult(as_frame=True)
        X = data.data
        y = (data.target == '>50K') * 1
        
        # Define privileged group for fairness calculations
        self.sex = X['sex']

        categorical_features = X.select_dtypes(include=['category', 'object']).columns
        numerical_features = X.select_dtypes(include=np.number).columns
        
        preprocessor = ColumnTransformer(transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

        X_processed = preprocessor.fit_transform(X)
        X_train, X_test, y_train, y_test, self.sex_train, self.sex_test = train_test_split(
            X_processed, y, self.sex, test_size=0.3, random_state=42
        )
        
        self.X_train_tensor = torch.tensor(X_train.toarray(), dtype=torch.float32)
        self.y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
        self.X_test_tensor = torch.tensor(X_test.toarray(), dtype=torch.float32)
        self.y_test = y_test.values

        # Create masks for fairness loss calculation
        self.privileged_mask_train = (self.sex_train == 'Male').values
        self.unprivileged_mask_train = (self.sex_train == 'Female').values

        logger.info("Data loaded successfully.")
        return X_train.shape[1]

    def _calculate_fairness_loss(self, y_pred_train):
        """Calculates a loss term based on demographic parity."""
        # Predictions for privileged group
        y_pred_priv = y_pred_train[self.privileged_mask_train]
        # Predictions for unprivileged group
        y_pred_unpriv = y_pred_train[self.unprivileged_mask_train]

        # The loss is the squared difference of the means of the predictions
        # for the two groups. We want these means to be as close as possible.
        if len(y_pred_priv) > 0 and len(y_pred_unpriv) > 0:
            mean_priv = torch.mean(y_pred_priv)
            mean_unpriv = torch.mean(y_pred_unpriv)
            return (mean_priv - mean_unpriv) ** 2
        return torch.tensor(0.0)

    def _evaluate_metrics(self):
        """Evaluates accuracy and fairness on the test set."""
        with torch.no_grad():
            y_pred_tensor = self.model(self.X_test_tensor)
            y_pred = (y_pred_tensor.numpy() > 0.5).astype(int).flatten()
        
        accuracy = np.mean(y_pred == self.y_test)
        fairness_dpd = demographic_parity_difference(self.y_test, y_pred, sensitive_features=self.sex_test)
        
        return {"accuracy": accuracy, "demographic_parity_difference": fairness_dpd}

    def _handle_commands(self):
        """Check for and handle commands from the server queues."""
        # Handle pause/resume
        try:
            cmd = self.server.messages_queue_by_type[PAUSE_RESUME_TYPE].get_nowait()
            if cmd.command == PAUSE_TRAINING:
                self.is_paused = True
                logger.info("Training paused.")
                self.server.update_server_state(cmd.__dict__)
            elif cmd.command == RESUME_TRAINING:
                self.is_paused = False
                logger.info("Training resumed.")
                self.server.update_server_state(cmd.__dict__)
        except queue.Empty:
            pass

        # Handle stop
        try:
            cmd = self.server.messages_queue_by_type[WRAPER_CONTROL_COMMAND_TYPE].get_nowait()
            if cmd.command == STOP_TRAINING:
                self.should_stop = True
                logger.info("Stopping training.")
        except queue.Empty:
            pass

        # Handle weight updates
        try:
            cmd = self.server.messages_queue_by_type[UPDATE_WEIGHTS_TYPE].get_nowait()
            if cmd.command == "update_weights":
                new_weights = json.loads(cmd.args)
                self.w_accuracy = new_weights.get("accuracy", self.w_accuracy)
                self.w_fairness = new_weights.get("fairness", self.w_fairness)
                logger.info(f"Updated weights: Accuracy={self.w_accuracy}, Fairness={self.w_fairness}")
                self.server.update_server_state(cmd.__dict__)
        except (queue.Empty, json.JSONDecodeError):
            pass

    def train(self, epochs=100, lr=0.01):

        # ---------------------------
        # Setup model and optimizer
        # ---------------------------
        n_features = self._load_and_preprocess_data()
        self.model = LogisticRegression(n_features)
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)

        logger.info("Starting interactive training (without energy monitoring)...")
        self.server.update_server_state({"command": RESUME_TRAINING})

        for epoch in range(epochs):

            # Handle interactive commands
            self._handle_commands()

            while self.is_paused and not self.should_stop:
                time.sleep(1)
                self._handle_commands()

            if self.should_stop:
                break

            # ---------------------------
            # Training step
            # ---------------------------
            self.optimizer.zero_grad()

            y_pred_train = self.model(self.X_train_tensor)

            accuracy_loss = self.criterion(
                y_pred_train,
                self.y_train_tensor
            )

            fairness_loss = self._calculate_fairness_loss(y_pred_train)

            total_loss = (
                self.w_accuracy * accuracy_loss +
                self.w_fairness * fairness_loss
            )

            total_loss.backward()
            self.optimizer.step()

            # ---------------------------
            # Evaluation
            # ---------------------------
            metrics = self._evaluate_metrics()

            # ---------------------------
            # Logging
            # ---------------------------
            log_data = {
                "epoch": epoch + 1,
                "total_loss": total_loss.item(),
                "accuracy_loss": accuracy_loss.item(),
                "fairness_loss": fairness_loss.item(),
                **metrics
            }

            log_event = {
                "command": "log_update",
                "args": json.dumps(log_data)
            }

            # Push to server
            self.server.set_latest_log(log_data)
            self.server.enqueue_event(log_event)

            logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Loss: {total_loss.item():.4f} | "
                f"Accuracy: {metrics['accuracy']:.4f} | "
                f"DPD: {metrics['demographic_parity_difference']:.4f}"
            )

        # ---------------------------
        # Training Finished
        # ---------------------------
        logger.info("Training finished.")

        self.server.update_server_state({
            "command": "stop_training",
            "status": "finished"
        })

        self.server.set_latest_log({"status": "finished"})


if __name__ == "__main__":
    HOST = "127.0.0.1"
    PORT = 8000
    
    server = InteractiveServer(host=HOST, port=PORT)
    server.run()
    logger.info(f"Interactive server running at http://{HOST}:{PORT}")
    
    trainer = InteractiveTrainer(server)
    
    training_thread = threading.Thread(target=trainer.train, daemon=True)
    training_thread.start()

    # training_thread.join()
    while True:
        time.sleep(1)
    logger.info("Application shutting down.")





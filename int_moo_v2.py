
import os
import uuid
import time
import json
import queue
import asyncio
import threading
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from fairlearn.datasets import fetch_adult
from fairlearn.metrics import demographic_parity_difference
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from uvicorn import Config, Server
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, APIRouter, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
UPDATE_WEIGHTS_TYPE = "update_weights_type"
PAUSE_RESUME_TYPE = "pause_resume_type"
WRAPPER_CONTROL_TYPE = "wrapper_control_command_type"

class CmdModel(BaseModel):
    model_id: str
    command: str
    args: str = "{}"

class LogisticRegression(nn.Module):
    def __init__(self, n_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_features, 1)
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

class InteractiveTrainer:
    def __init__(self, model_id: str, server: 'InteractiveServer'):
        self.model_id = model_id
        self.server = server
        self.status = "init"
        self.is_paused = False
        self.should_stop = False
        self.w_accuracy = 1.0
        self.w_fairness = 0.5
        self.latest_log = {}
        self.queues = {
            UPDATE_WEIGHTS_TYPE: queue.Queue(),
            PAUSE_RESUME_TYPE: queue.Queue(),
            WRAPPER_CONTROL_TYPE: queue.Queue(),
        }

    def _load_data(self):
        data = fetch_adult(as_frame=True)
        X, y = data.data, (data.target == '>50K') * 1
        sex = X['sex']
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), X.select_dtypes(include=np.number).columns),
            ('cat', OneHotEncoder(handle_unknown='ignore'), X.select_dtypes(include=['category', 'object']).columns)
        ])
        X_p = preprocessor.fit_transform(X)
        X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(X_p, y, sex, test_size=0.3)
        
        self.X_train_t = torch.tensor(X_train.toarray(), dtype=torch.float32)
        self.y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
        self.X_test_t = torch.tensor(X_test.toarray(), dtype=torch.float32)
        self.y_test = y_test.values
        self.sex_test = s_test
        self.priv_mask = (s_train == 'Male').values
        self.unpriv_mask = (s_train == 'Female').values
        return X_train.shape[1]

    def train(self):
        n_features = self._load_data()
        model = LogisticRegression(n_features)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.BCELoss()
        self.status = "running"

        for epoch in range(500):
            # Handle Commands
            for q_type in self.queues:
                try:
                    while not self.queues[q_type].empty():
                        cmd = self.queues[q_type].get_nowait()
                        if cmd['command'] == "pause_training": self.is_paused = True
                        if cmd['command'] == "resume_training": self.is_paused = False
                        if cmd['command'] == "stop_training": self.should_stop = True
                        if cmd['command'] == "update_weights":
                            weights = json.loads(cmd['args'])
                            self.w_accuracy = weights.get("accuracy", self.w_accuracy)
                            self.w_fairness = weights.get("fairness", self.w_fairness)
                except queue.Empty: pass

            while self.is_paused and not self.should_stop:
                self.status = "paused"
                time.sleep(0.5)
            if self.should_stop: break
            
            self.status = "running"
            optimizer.zero_grad()
            outputs = model(self.X_train_t)
            loss_acc = criterion(outputs, self.y_train_t)
            
            # Fairness Loss
            p_mean, u_mean = torch.mean(outputs[self.priv_mask]), torch.mean(outputs[self.unpriv_mask])
            loss_fair = (p_mean - u_mean)**2
            
            total_loss = (self.w_accuracy * loss_acc) + (self.w_fairness * loss_fair)
            total_loss.backward()
            optimizer.step()

            # Eval
            with torch.no_grad():
                preds = (model(self.X_test_t).numpy() > 0.5).astype(int).flatten()
                acc = np.mean(preds == self.y_test)
                dpd = demographic_parity_difference(self.y_test, preds, sensitive_features=self.sex_test)

            self.latest_log = {"epoch": epoch+1, "accuracy": acc, "fairness": dpd, "loss": total_loss.item()}
            self.server.enqueue_event({"model_id": self.model_id, "data": self.latest_log})
            time.sleep(0.1) # Simulate work

        self.status = "finished"

class InteractiveServer:
    def __init__(self, host: str, port: int):
        self.app = FastAPI()
        self.host, self.port = host, port
        self.trainers: Dict[str, InteractiveTrainer] = {}
        self.events_queue = queue.Queue()
        self.app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
        self._setup_routes()

    def _setup_routes(self):
        @self.app.get("/api/models")
        async def list_models():
            return {mid: {"status": t.status, "weights": {"acc": t.w_accuracy, "fair": t.w_fairness}} 
                    for mid, t in self.trainers.items()}

        @self.app.post("/api/create/{model_id}")
        async def create_trainer(model_id: str, background_tasks: BackgroundTasks):
            if model_id not in self.trainers:
                trainer = InteractiveTrainer(model_id, self)
                self.trainers[model_id] = trainer
                background_tasks.add_task(trainer.train)
                return {"status": "created"}
            return {"status": "exists"}

        @self.app.get("/api/logs/{model_id}")
        async def get_log(model_id: str):
            return self.trainers.get(model_id).latest_log if model_id in self.trainers else {}

        @self.app.post("/api/command/")
        async def receive_command(cmd: CmdModel):
            if cmd.model_id in self.trainers:
                t = self.trainers[cmd.model_id]
                cmd_dict = cmd.model_dump()
                # Route to correct queue
                if cmd.command in ["pause_training", "resume_training"]:
                    t.queues[PAUSE_RESUME_TYPE].put(cmd_dict)
                elif cmd.command == "update_weights":
                    t.queues[UPDATE_WEIGHTS_TYPE].put(cmd_dict)
                elif cmd.command in ["stop_training", "start_training"]:
                    t.queues[WRAPPER_CONTROL_TYPE].put(cmd_dict)
                return {"status": "queued"}
            return {"status": "model_not_found"}

    def enqueue_event(self, event):
        self.events_queue.put(event)

    def run(self):
        config = Config(app=self.app, host=self.host, port=self.port, loop="asyncio")
        server = Server(config=config)
        server.run()

if __name__ == "__main__":
    server = InteractiveServer("127.0.0.1", 8000)
    server.run()
import time
import json
import queue
import threading
import os
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict
from fairlearn.datasets import fetch_adult
from fairlearn.metrics import demographic_parity_difference
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from uvicorn import Config, Server
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from codecarbon import EmissionsTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for command routing
UPDATE_WEIGHTS_TYPE = "update_weights_type"
PAUSE_RESUME_TYPE = "pause_resume_type"
CONTROL_TYPE = "control_type"

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
    def __init__(self, model_id: str, server: 'InteractiveServer', total_epochs: int):
        self.model_id = model_id
        self.server = server
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.status = "init"
        self.is_paused = False
        self.should_stop = False
        self.start_event = threading.Event() # New: cleaner start signal
        self.w_accuracy = 1.0
        self.w_fairness = 0.5
        self.w_energy = 0.5
        self.latest_log = {}
        self.tracker = None
        # Queues for thread-safe communication
        self.queues = {
            UPDATE_WEIGHTS_TYPE: queue.Queue(),
            PAUSE_RESUME_TYPE: queue.Queue(),
            CONTROL_TYPE: queue.Queue(),
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
        try:
            n_features = self._load_data()
            model = LogisticRegression(n_features)
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            criterion = nn.BCELoss()
            
            # Fix: removing logging_level to avoid unknown keyword argument error
            self.tracker = EmissionsTracker(project_name=f"Interactive_{self.model_id}", 
                                           measure_power_secs=1,
                                           save_to_file=False)
            self.tracker.start()
            
            self.status = "init"

            # Wait for manual start event
            while not self.start_event.is_set() and not self.should_stop:
                time.sleep(0.5)

            if self.should_stop: 
                if self.tracker: self.tracker.stop()
                return

            self.status = "running"
            for epoch in range(self.total_epochs):
                self.current_epoch = epoch + 1
                
                # Check for commands in queues
                for q in self.queues.values():
                    while not q.empty():
                        cmd = q.get_nowait()
                        if cmd['command'] == "pause_training": self.is_paused = True
                        elif cmd['command'] == "resume_training": self.is_paused = False
                        elif cmd['command'] == "stop_training": self.should_stop = True
                        elif cmd['command'] == "update_weights":
                            w = json.loads(cmd['args'])
                            self.w_accuracy = w.get('accuracy', self.w_accuracy)
                            self.w_fairness = w.get('fairness', self.w_fairness)
                            self.w_energy = w.get('energy', self.w_energy)

                while self.is_paused and not self.should_stop:
                    self.status = "paused"
                    time.sleep(0.5)
                if self.should_stop: break
                
                self.status = "running"
                optimizer.zero_grad()
                outputs = model(self.X_train_t)
                loss_acc = criterion(outputs, self.y_train_t)
                
                # Demographic Parity Loss (Squared Difference of Means)
                p_mean = torch.mean(outputs[self.priv_mask])
                u_mean = torch.mean(outputs[self.unpriv_mask])
                loss_fair = (p_mean - u_mean)**2
                bias = float((p_mean - u_mean).detach().item())
                
                # Energy Efficiency Proxy (L2 weight penalty)
                l2_reg = torch.tensor(0.)
                for param in model.parameters():
                    l2_reg += torch.norm(param)
                loss_energy = l2_reg
                
                total_loss = (self.w_accuracy * loss_acc) + (self.w_fairness * loss_fair) + (self.w_energy * loss_energy)
                total_loss.backward()
                optimizer.step()

                # Evaluation
                with torch.no_grad():
                    preds = (model(self.X_test_t).numpy() > 0.5).astype(int).flatten()
                    acc = np.mean(preds == self.y_test)
                    dpd = demographic_parity_difference(self.y_test, preds, sensitive_features=self.sex_test)

                self.latest_log = {
                    "epoch": self.current_epoch, 
                    "total_epochs": self.total_epochs,
                    "accuracy": float(acc), 
                    "fairness": float(dpd), 
                    "bias": bias,
                    "loss": float(total_loss.item()),
                    "energy_consumed": float(self.tracker._total_energy.kWh),
                    "power_draw": float(sum(self.tracker._get_power().values())) if self.tracker._get_power() else 0.0
                }
                time.sleep(0.05) 

            self.status = "finished"
            if self.tracker:
                self.tracker.stop()
        except Exception as e:
            logger.error(f"Error in trainer {self.model_id}: {e}")
            self.status = "error"

class InteractiveServer:
    def __init__(self, host: str, port: int):
        self.app = FastAPI()
        self.host, self.port = host, port
        self.trainers: Dict[str, InteractiveTrainer] = {}
        self.app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
        self._setup_routes()

    def _setup_routes(self):
        @self.app.get("/api/models")
        async def list_models():
            return {mid: {"status": t.status, "epoch": t.current_epoch, "total": t.total_epochs} 
                    for mid, t in self.trainers.items()}

        @self.app.post("/api/create/{model_id}/{epochs}")
        async def create_trainer(model_id: str, epochs: int, background_tasks: BackgroundTasks):
            if model_id in self.trainers and self.trainers[model_id].status != "finished":
                return {"status": "already_running"}
            
            trainer = InteractiveTrainer(model_id, self, epochs)
            self.trainers[model_id] = trainer
            background_tasks.add_task(trainer.train)
            return {"status": "started", "model_id": model_id, "epochs": epochs}

        @self.app.get("/api/logs/{model_id}")
        async def get_log(model_id: str):
            if model_id in self.trainers:
                return self.trainers[model_id].latest_log
            return {}

        @self.app.post("/api/command/")
        async def receive_command(cmd: CmdModel):
            if cmd.model_id in self.trainers:
                t = self.trainers[cmd.model_id]
                if cmd.command in ["pause_training", "resume_training"]:
                    t.queues[PAUSE_RESUME_TYPE].put(cmd.model_dump())
                elif cmd.command == "update_weights":
                    t.queues[UPDATE_WEIGHTS_TYPE].put(cmd.model_dump())
                elif cmd.command == "start_training":
                    t.start_event.set()
                elif cmd.command == "stop_training":
                    t.should_stop = True
                return {"status": "command_sent"}
            return {"status": "not_found"}

        @self.app.delete("/api/delete/{model_id}")
        async def delete_trainer(model_id: str):
            if model_id in self.trainers:
                trainer = self.trainers[model_id]
                trainer.should_stop = True
                trainer.start_event.set() # Release if waiting
                del self.trainers[model_id]
                return {"status": "deleted"}
            return {"status": "not_found", "model_id": model_id}

    def run(self):
        config = Config(app=self.app, host=self.host, port=self.port, loop="asyncio")
        Server(config=config).run()

def kill_process_on_port(port):
    try:
        # Get the PID of the process using the port
        result = subprocess.check_output(f"netstat -ano | findstr LISTENING | findstr :{port}", shell=True).decode()
        if result:
            lines = result.strip().split('\n')
            for line in lines:
                parts = line.split()
                if len(parts) > 4:
                    pid = parts[-1]
                    logger.info(f"Releasing port {port} (Terminating PID: {pid})...")
                    os.system(f"taskkill /F /PID {pid}")
    except subprocess.CalledProcessError:
        # Port is likely already free
        pass
    except Exception as e:
        logger.error(f"Error releasing port {port}: {e}")

if __name__ == "__main__":
    target_port = 5000
    kill_process_on_port(target_port)
    server = InteractiveServer("127.0.0.1", target_port)
    server.run()

import time
import queue
import threading
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from fairlearn.datasets import fetch_adult
from fairlearn.metrics import demographic_parity_difference
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from libraries.power_tracker import DynamicPowerTracker
from models import LogisticRegression, DeepNeuralNetwork

logger = logging.getLogger(__name__)

UPDATE_WEIGHTS_TYPE = "update_weights_type"
PAUSE_RESUME_TYPE = "pause_resume_type"

class InteractiveTrainer:
    def __init__(self, model_id: str, model_type: str, total_epochs: int):
        self.model_id = model_id
        self.model_type = model_type  
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.status = "init"
        self.is_paused = False
        self.should_stop = False
        self.start_event = threading.Event()
        self.w_accuracy = 1.0
        self.w_fairness = 0.5
        self.w_energy = 0.5
        self.latest_log = {}
        self.tracker = None
        
        self.queues = {
            UPDATE_WEIGHTS_TYPE: queue.Queue(),
            PAUSE_RESUME_TYPE: queue.Queue(),
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
            
            if self.model_type == 'dnn':
                model = DeepNeuralNetwork(n_features)
            else:
                model = LogisticRegression(n_features)
                
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            criterion = nn.BCELoss()
            
            self.tracker = DynamicPowerTracker(self.model_id)
            self.tracker.start()
            
            self.status = "init"

            while not self.start_event.is_set() and not self.should_stop:
                time.sleep(0.5)

            if self.should_stop: 
                if self.tracker: self.tracker.stop()
                return

            self.status = "running"
            for epoch in range(self.total_epochs):
                self.current_epoch = epoch + 1
                
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
                
                p_mean = torch.mean(outputs[self.priv_mask])
                u_mean = torch.mean(outputs[self.unpriv_mask])
                loss_fair = (p_mean - u_mean)**2
                bias = float((p_mean - u_mean).detach().item())
                
                l2_reg = torch.tensor(0.)
                for param in model.parameters():
                    l2_reg += torch.norm(param)
                loss_energy = l2_reg
                
                total_loss = (self.w_accuracy * loss_acc) + (self.w_fairness * loss_fair) + (self.w_energy * loss_energy)
                total_loss.backward()
                optimizer.step()

                with torch.no_grad():
                    preds = (model(self.X_test_t).numpy() > 0.5).astype(int).flatten()
                    acc = np.mean(preds == self.y_test)
                    try:
                        dpd = demographic_parity_difference(self.y_test, preds, sensitive_features=self.sex_test)
                    except:
                        dpd = 0.0

                power_w, energy_kwh = self.tracker.get_power_energy()

                self.latest_log = {
                    "epoch": self.current_epoch, 
                    "total_epochs": self.total_epochs,
                    "model_type": self.model_type,
                    "accuracy": float(acc), 
                    "fairness": float(dpd), 
                    "bias": float(bias),
                    "loss": float(total_loss.item()),
                    "energy_consumed": float(energy_kwh),
                    "power_draw": float(power_w)
                }
                time.sleep(0.05) 

            self.status = "finished"
            if self.tracker:
                self.tracker.stop()
        except Exception as e:
            logger.error(f"Error in trainer {self.model_id}: {e}")
            self.status = "error"

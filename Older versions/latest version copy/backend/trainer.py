import time
import queue
import threading
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from fairlearn.metrics import demographic_parity_difference

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from libraries.power_tracker import DynamicPowerTracker
from models import LogisticRegression, DeepNeuralNetwork, ResNet18Custom
from data_loader import get_adult_dataloaders, get_synthetic_vision_dataloaders

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

    def train(self):
        try:
            if self.model_type == 'resnet':
                train_loader, test_data, _ = get_synthetic_vision_dataloaders()
                model = ResNet18Custom()
            elif self.model_type == 'dnn':
                train_loader, test_data, n_features = get_adult_dataloaders()
                model = DeepNeuralNetwork(n_features)
            else:
                train_loader, test_data, n_features = get_adult_dataloaders()
                model = LogisticRegression(n_features)
                
            self.X_test_t = test_data['X_test_t']
            self.y_test = test_data['y_test']
            self.sex_test = test_data['sex_test']
                
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            criterion = nn.BCELoss()
            
            self.tracker = DynamicPowerTracker(self.model_id)
            self.tracker.start()
            
            self.status = "init"
            
            start_time = time.time()
            total_elapsed_time = 0.0

            while not self.start_event.is_set() and not self.should_stop:
                time.sleep(0.5)

            if self.should_stop: 
                if self.tracker: self.tracker.stop()
                return

            self.status = "running"
            for epoch in range(self.total_epochs):
                self.current_epoch = epoch + 1
                
                def process_queues():
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

                process_queues()

                while self.is_paused and not self.should_stop:
                    self.status = "paused"
                    time.sleep(0.5)
                    start_time += 0.5 # Shift start time forward so pause doesn't add to elapsed
                    process_queues()
                if self.should_stop: break
                
                self.status = "running"
                
                total_loss_val = 0.0
                total_bias_val = 0.0
                
                model.train()
                for batch_idx, (batch_X, batch_y, batch_priv, batch_unpriv) in enumerate(train_loader):
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss_acc = criterion(outputs, batch_y)
                    
                    if batch_priv.any() and batch_unpriv.any():
                        p_mean = torch.mean(outputs[batch_priv])
                        u_mean = torch.mean(outputs[batch_unpriv])
                        loss_fair = (p_mean - u_mean)**2
                        batch_bias = float((p_mean - u_mean).detach().item())
                    else:
                        loss_fair = torch.tensor(0.0)
                        batch_bias = 0.0
                        
                    l2_reg = torch.tensor(0.)
                    for param in model.parameters():
                        l2_reg += torch.norm(param)
                    loss_energy = l2_reg
                    
                    batch_loss = (self.w_accuracy * loss_acc) + (self.w_fairness * loss_fair) + (self.w_energy * loss_energy)
                    batch_loss.backward()
                    optimizer.step()
                    
                    total_loss_val += batch_loss.item()
                    total_bias_val += batch_bias
                    
                    # Update elapsed time
                    if not self.is_paused:
                        total_elapsed_time = time.time() - start_time
                        
                    # Calculate intermittent power draw
                    power_w, energy_kwh = self.tracker.get_power_energy()

                    self.latest_log = {
                        "epoch": self.current_epoch, 
                        "total_epochs": self.total_epochs,
                        "iteration": batch_idx + 1,
                        "total_iterations": len(train_loader),
                        "model_type": self.model_type,
                        "accuracy": getattr(self, "last_acc", 0.0), # Updated below
                        "fairness": getattr(self, "last_dpd", 0.0),
                        "bias": float(batch_bias),
                        "loss": float(batch_loss.item()),
                        "energy_consumed": float(energy_kwh),
                        "power_draw": float(power_w),
                        "elapsed_time": float(total_elapsed_time)
                    }

                model.eval()
                with torch.no_grad():
                    preds = []
                    chunk_size = 128
                    for i in range(0, len(self.X_test_t), chunk_size):
                        batch = self.X_test_t[i:i+chunk_size]
                        out = model(batch)
                        preds.extend((out.numpy() > 0.5).astype(int).flatten())
                    preds = np.array(preds)
                    
                    acc = np.mean(preds == self.y_test)
                    try:
                        dpd = demographic_parity_difference(self.y_test, preds, sensitive_features=self.sex_test)
                    except:
                        dpd = 0.0
                        
                    # Save for next iterations
                    self.last_acc = float(acc)
                    self.last_dpd = float(dpd)

                power_w, energy_kwh = self.tracker.get_power_energy()

                self.latest_log = {
                    "epoch": self.current_epoch, 
                    "total_epochs": self.total_epochs,
                    "iteration": "Done",
                    "total_iterations": len(train_loader),
                    "model_type": self.model_type,
                    "accuracy": float(acc), 
                    "fairness": float(dpd), 
                    "bias": float(total_bias_val / len(train_loader)),
                    "loss": float(total_loss_val / len(train_loader)),
                    "energy_consumed": float(energy_kwh),
                    "power_draw": float(power_w),
                    "elapsed_time": float(total_elapsed_time)
                }
                time.sleep(0.05) 

            self.status = "finished"
            if self.tracker:
                self.tracker.stop()
        except Exception as e:
            logger.error(f"Error in trainer {self.model_id}: {e}")
            self.status = "error"

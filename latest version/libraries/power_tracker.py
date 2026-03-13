import platform
import subprocess
import time
import logging

try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger(__name__)

class DynamicPowerTracker:
    def __init__(self, model_id):
        self.model_id = model_id
        self.cpu_name = platform.processor() or "Unknown CPU"
        self.gpu_name = "Unknown GPU"
        self.cpu_tdp = 65.0  # Default fallback W
        self.gpu_tdp = 100.0 # Default fallback W
        self.is_running = False
        self.total_energy_kwh = 0.0
        self.current_power_w = 0.0
        self.last_time = time.time()
        self._detect_hardware()

    def _detect_hardware(self):
        try:
            cpu_raw = subprocess.check_output("wmic cpu get name", shell=True).decode().strip().split('\n')
            if len(cpu_raw) > 1:
                self.cpu_name = cpu_raw[1].strip()
                if "Ryzen 5" in self.cpu_name or "Ryzen 7" in self.cpu_name:
                    self.cpu_tdp = 45.0
                elif "Ryzen 9" in self.cpu_name:
                    self.cpu_tdp = 65.0
            
            gpu_raw = subprocess.check_output("wmic path win32_VideoController get name", shell=True).decode().strip().split('\n')
            for line in gpu_raw[1:]:
                gpu = line.strip()
                if gpu and "Radeon" in gpu and "RX" in gpu:
                    self.gpu_name = gpu
                    self.gpu_tdp = 100.0
                    break
            
            logger.info(f"[{self.model_id}] Telemetry - CPU: {self.cpu_name} (Est. {self.cpu_tdp}W), GPU: {self.gpu_name} (Est. {self.gpu_tdp}W)")
        except Exception as e:
            logger.warning(f"Hardware detection warning: {e}")

    def start(self):
        self.is_running = True
        self.last_time = time.time()
        if psutil: psutil.cpu_percent()

    def stop(self):
        self.is_running = False

    def get_power_energy(self):
        if not self.is_running:
            return self.current_power_w, self.total_energy_kwh
            
        now = time.time()
        delta_h = (now - self.last_time) / 3600.0
        self.last_time = now
        
        try:
            cpu_util = psutil.cpu_percent() / 100.0 if psutil else 0.5
        except:
            cpu_util = 0.5
            
        gpu_util = 0.8 # Proxy for heavy PyTorch load
        
        self.current_power_w = (cpu_util * self.cpu_tdp) + (gpu_util * self.gpu_tdp)
        self.total_energy_kwh += (self.current_power_w / 1000.0) * delta_h
        
        return self.current_power_w, self.total_energy_kwh

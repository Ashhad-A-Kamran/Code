import logging
import os
import subprocess
from uvicorn import Config, Server
from api import InteractiveServer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def kill_process_on_port(port):
    try:
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
        pass
    except Exception as e:
        logger.error(f"Error releasing port {port}: {e}")

if __name__ == "__main__":
    target_port = 5000
    kill_process_on_port(target_port)
    server_instance = InteractiveServer("127.0.0.1", target_port)
    config = Config(app=server_instance.app, host=server_instance.host, port=server_instance.port, loop="asyncio")
    Server(config=config).run()

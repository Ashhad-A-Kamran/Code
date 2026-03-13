import os
from typing import Dict
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import logging

from trainer import InteractiveTrainer, UPDATE_WEIGHTS_TYPE, PAUSE_RESUME_TYPE

logger = logging.getLogger(__name__)

class CmdModel(BaseModel):
    model_id: str
    command: str
    args: str = "{}"

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
            return {mid: {
                "status": t.status, 
                "epoch": t.current_epoch, 
                "total": t.total_epochs,
                "type": t.model_type
            } for mid, t in self.trainers.items()}

        @self.app.post("/api/create/{model_id}/{model_type}/{epochs}")
        async def create_trainer(model_id: str, model_type: str, epochs: int, background_tasks: BackgroundTasks):
            if model_id in self.trainers and self.trainers[model_id].status not in ["finished", "error"]:
                return {"status": "already_running"}
            
            trainer = InteractiveTrainer(model_id, model_type, epochs)
            self.trainers[model_id] = trainer
            background_tasks.add_task(trainer.train)
            return {"status": "started", "model_id": model_id, "type": model_type, "epochs": epochs}

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
                trainer.start_event.set()
                del self.trainers[model_id]
                return {"status": "deleted"}
            return {"status": "not_found", "model_id": model_id}

        ui_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "web_ui_v5"))
        self.app.mount("/static", StaticFiles(directory=ui_path), name="static")

        @self.app.get("/")
        async def serve_index():
            return FileResponse(os.path.join(ui_path, "index.html"))

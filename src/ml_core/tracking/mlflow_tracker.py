import mlflow
import mlflow.pytorch
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any


class MLflowTracker:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # lokaal, file-based (Snellius-proof)
        mlflow.set_tracking_uri("file:/scratch-shared/scur2378/mlruns")
        mlflow.set_experiment("pcam_assignment_2")

        self.run = mlflow.start_run()

        # log volledige config
        mlflow.log_params(self._flatten_dict(config))

        # log git commit
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"]
        ).decode("utf-8").strip()
        mlflow.log_param("git_commit", git_commit)

        # log environment
        reqs = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"]
        ).decode("utf-8")

        Path("requirements_logged.txt").write_text(reqs)
        mlflow.log_artifact("requirements_logged.txt")

    def log_metrics(self, metrics: Dict[str, float], step: int):
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)

    def log_model(self, model, artifact_path: str):
        mlflow.pytorch.log_model(model, artifact_path)

    def end(self):
        mlflow.end_run()

    def _flatten_dict(self, d, parent_key="", sep="."):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

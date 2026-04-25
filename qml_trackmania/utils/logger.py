"""
utils/logger.py
===============
Simple training logger that writes CSV metrics and optionally syncs to W&B.
"""

import os
import csv
import time
from typing import Optional, Dict
import logging

log = logging.getLogger(__name__)


class TrainingLogger:
    """
    Logs training metrics to:
      - Console
      - CSV file (always)
      - Weights & Biases (if cfg.log_wandb=True)
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.csv_path = f"{cfg.run_name}_metrics.csv"
        self._init_csv()

        self.wandb_run = None
        if cfg.log_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=cfg.wandb_project,
                    name=cfg.run_name,
                    config=vars(cfg),
                )
                log.info(f"W&B run initialised: {self.wandb_run.url}")
            except Exception as e:
                log.warning(f"W&B init failed: {e}. Logging to CSV only.")

    def _init_csv(self):
        os.makedirs(os.path.dirname(self.csv_path) if os.path.dirname(self.csv_path) else ".", exist_ok=True)
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "step", "episode_reward", "episode_length",
                "critic_loss", "actor_loss", "alpha_loss", "alpha", "timestamp"
            ])

    def log_episode(self, step: int, reward: float, length: int, losses: Dict):
        row = {
            "step":           step,
            "episode_reward": reward,
            "episode_length": length,
            "critic_loss":    losses.get("critic_loss", 0.0),
            "actor_loss":     losses.get("actor_loss", 0.0),
            "alpha_loss":     losses.get("alpha_loss", 0.0),
            "alpha":          losses.get("alpha", 0.0),
            "timestamp":      time.time(),
        }

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writerow(row)

        if self.wandb_run:
            self.wandb_run.log(row, step=step)

    def close(self):
        if self.wandb_run:
            self.wandb_run.finish()
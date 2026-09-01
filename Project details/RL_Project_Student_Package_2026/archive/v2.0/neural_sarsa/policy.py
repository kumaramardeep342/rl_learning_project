# Neural Network based SARSA Policy
# Name : Amardeep Kumar
# Roll No : DA25M502

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

MODEL_PATH = Path(__file__).parent / "neural_sarsa_weights.pt"
checkpoint = torch.load(MODEL_PATH, map_location=torch.device("cpu"))

STATE_DIM = checkpoint["state_dim"]
ACTION_DIM = checkpoint["action_dim"]
ALLOWED_ACTIONS = checkpoint["actions"]

class DeepSARSAQNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

MODEL = DeepSARSAQNet(STATE_DIM, ACTION_DIM)
MODEL.load_state_dict(checkpoint["model_state"])
MODEL.eval()

def _preprocess_obs(obs: dict) -> np.ndarray:
    inv = np.asarray(obs["inventory"], dtype=np.float32).flatten()
    pipeline = np.asarray(obs["arrival_pipeline"], dtype=np.float32)
    inv_pos = (inv + pipeline.sum(axis=1)) / 300.0
    inv_norm = inv / 200.0
    pipe_norm = pipeline.flatten() / 100.0
    demand_hist = np.asarray(obs["demand_history"], dtype=np.float32)
    d_mean_7 = demand_hist.mean(axis=0) / 50.0
    d_mean_3 = demand_hist[-3:].mean(axis=0) / 50.0
    day_norm = np.asarray([obs["day"]], dtype=np.float32).flatten() / 50.0
    cap_util = np.asarray([obs["capacity_utilisation"]], dtype=np.float32).flatten()
    return np.concatenate([inv_pos, inv_norm, pipe_norm, d_mean_7, d_mean_3, day_norm, cap_util])

def run_policy(observation):
    state = _preprocess_obs(observation)
    with torch.no_grad():
        s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_idx = int(MODEL(s_t).argmax(dim=1).item())
    return [int(q) for q in ALLOWED_ACTIONS[action_idx]]
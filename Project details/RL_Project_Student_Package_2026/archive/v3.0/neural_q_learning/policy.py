# Neural Network based Q-Learning Policy
# Name : Amardeep Kumar
# Roll No : DA25M502

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

MODEL_PATH = Path(__file__).parent / "neural_q_weights.pt"
checkpoint = torch.load(MODEL_PATH, map_location=torch.device("cpu"))

STATE_DIM = checkpoint["state_dim"]
ACTION_DIM = checkpoint["action_dim"]
ALLOWED_ACTIONS = checkpoint["actions"]

class NeuralQNet(nn.Module):
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

MODEL = NeuralQNet(STATE_DIM, ACTION_DIM)
MODEL.load_state_dict(checkpoint["model_state"])
MODEL.eval()

def _preprocess_obs(obs: dict) -> np.ndarray:
    inv = np.asarray(obs["inventory"], dtype=np.float32).flatten()
    pipeline = np.asarray(obs["arrival_pipeline"], dtype=np.float32)

    inv_pos = inv + pipeline.sum(axis=1)
    mean_demands = np.array([30.0, 25.0, 35.0], dtype=np.float32)
    days_of_supply = (inv_pos / mean_demands) / 5.0

    current_vol = 2.0 * inv[0] + 3.0 * inv[1] + 1.5 * inv[2]
    vol_slack = (1000.0 - current_vol) / 1000.0

    pipe_norm = pipeline.flatten() / 100.0
    demand_hist = np.asarray(obs["demand_history"], dtype=np.float32)
    d_mean_3 = demand_hist[-3:].mean(axis=0) / 50.0
    d_mean_7 = demand_hist.mean(axis=0) / 50.0
    day_norm = np.asarray([obs["day"]], dtype=np.float32).flatten() / 50.0

    return np.concatenate([
        days_of_supply,
        inv / 200.0,
        [vol_slack],
        pipe_norm,
        d_mean_3,
        d_mean_7,
        day_norm
    ])

def run_policy(observation):
    state = _preprocess_obs(observation)
    with torch.no_grad():
        s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_idx = int(MODEL(s_t).argmax(dim=1).item())
    return [int(q) for q in ALLOWED_ACTIONS[action_idx]]
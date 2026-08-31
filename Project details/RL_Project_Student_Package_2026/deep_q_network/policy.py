# Deep Q-Network Policy
# Name : Amardeep Kumar
# Roll No : DA25M502

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

# Load checkpoint on CPU for fast portable inference
MODEL_PATH = Path(__file__).parent / "dqn_weights.pt"
checkpoint = torch.load(MODEL_PATH, map_location=torch.device("cpu"))

STATE_DIM = checkpoint["state_dim"]
ACTION_DIM = checkpoint["action_dim"]
ALLOWED_ACTIONS = checkpoint["actions"]


class QNetwork(nn.Module):

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


MODEL = QNetwork(STATE_DIM, ACTION_DIM)
MODEL.load_state_dict(checkpoint["model_state"])
MODEL.eval()


def _preprocess_obs(obs: dict) -> np.ndarray:
    inv = np.asarray(obs["inventory"], dtype=np.float32).flatten() / 200.0
    pipeline = (
        np.asarray(obs["arrival_pipeline"], dtype=np.float32).flatten() / 100.0
    )
    demand = (
        np.asarray(obs["demand_history"], dtype=np.float32).flatten() / 50.0
    )
    day = np.asarray([obs["day"]], dtype=np.float32).flatten() / 50.0
    util = np.asarray(
        [obs["capacity_utilisation"]], dtype=np.float32
    ).flatten()
    return np.concatenate([inv, pipeline, demand, day, util])


def run_policy(observation):
    """Deterministic inference for DQN."""
    state = _preprocess_obs(observation)
    with torch.no_grad():
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_idx = int(MODEL(state_t).argmax(dim=1).item())

    return [int(q) for q in ALLOWED_ACTIONS[action_idx]]
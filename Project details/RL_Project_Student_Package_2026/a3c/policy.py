# A3C Policy
# Name : Amardeep Kumar
# Roll No : DA25M502

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

MODEL_PATH = Path(__file__).parent / "a3c_weights.pt"
checkpoint = torch.load(MODEL_PATH, map_location=torch.device("cpu"))

STATE_DIM = checkpoint["state_dim"]
ACTION_DIM = checkpoint["action_dim"]
ALLOWED_ACTIONS = checkpoint["actions"]


class ActorCriticA3C(nn.Module):

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
        )
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor):
        feat = self.shared(x)
        logits = self.actor(feat)
        val = self.critic(feat)
        return logits, val


MODEL = ActorCriticA3C(STATE_DIM, ACTION_DIM)
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
    """Deterministic inference for A3C."""
    state = _preprocess_obs(observation)
    with torch.no_grad():
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        logits, _ = MODEL(state_t)
        action_idx = int(logits.argmax(dim=1).item())

    return [int(q) for q in ALLOWED_ACTIONS[action_idx]]
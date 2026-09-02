# TD(𝜆) with Eligibility Traces Policy
# Name : Amardeep Kumar
# Roll No : DA25M502 

from pathlib import Path
import pickle
import numpy as np

MODEL_PATH = Path(__file__).parent / "td_lambda_table.pkl"
with open(MODEL_PATH, "rb") as f:
    DATA = pickle.load(f)

Q_TABLE = DATA["q_table"]
ALLOWED_ACTIONS = DATA["actions"]

def _get_state(obs: dict) -> tuple:
    inv = np.asarray(obs["inventory"], dtype=float)
    pipeline = np.asarray(obs["arrival_pipeline"], dtype=float)
    inv_pos = inv + pipeline.sum(axis=1)

    bins_p1 = [0, 40, 70, 95, 120, 150, 180, 220]
    bins_p2 = [0, 30, 55, 80, 105, 130, 160, 200]
    bins_p3 = [0, 40, 75, 105, 135, 170, 200, 240]

    b1 = int(np.digitize(inv_pos[0], bins_p1))
    b2 = int(np.digitize(inv_pos[1], bins_p2))
    b3 = int(np.digitize(inv_pos[2], bins_p3))
    return (b1, b2, b3)

def run_policy(observation):
    """Deterministic entrypoint for TD(lambda) inference."""
    state = _get_state(observation)
    if state in Q_TABLE:
        action_idx = int(np.argmax(Q_TABLE[state]))
    else:
        action_idx = 1  # Base replenishment default
    return [int(q) for q in ALLOWED_ACTIONS[action_idx]]
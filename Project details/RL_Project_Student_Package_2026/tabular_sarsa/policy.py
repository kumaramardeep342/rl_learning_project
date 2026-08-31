# Tabular Q-Sarsa Policy
# Name : Amardeep Kumar
# Roll No : DA25M502

from pathlib import Path
import pickle
import numpy as np

# Load trained SARSA table parameters
MODEL_PATH = Path(__file__).parent / "sarsa_table.pkl"

with open(MODEL_PATH, "rb") as f:
    DATA = pickle.load(f)

Q_TABLE = DATA["q_table"]
ALLOWED_ACTIONS = DATA["actions"]


def _get_state(obs: dict) -> tuple:
    inv = np.asarray(obs["inventory"], dtype=float)
    pipeline = np.asarray(obs["arrival_pipeline"], dtype=float)
    inv_pos = inv + pipeline.sum(axis=1)

    bins = [0, 40, 70, 100, 130, 160, 200]
    b1 = int(np.digitize(inv_pos[0], bins))
    b2 = int(np.digitize(inv_pos[1], bins))
    b3 = int(np.digitize(inv_pos[2], bins))
    return (b1, b2, b3)


def run_policy(observation):
    """Deterministic inference for Tabular SARSA."""
    state = _get_state(observation)

    if state in Q_TABLE:
        action_idx = int(np.argmax(Q_TABLE[state]))
    else:
        action_idx = 0

    return [int(q) for q in ALLOWED_ACTIONS[action_idx]]
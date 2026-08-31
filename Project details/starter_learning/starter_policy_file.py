# starter_policy_file.py
def run_policy(observation):
    inv = observation["inventory"]
    q1 = 20 if inv[0] < 50 else 0
    q2 = 20 if inv[1] < 40 else 0
    q3 = 10 if inv[2] < 30 else 0
    return [q1, q2, q3]
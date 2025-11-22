# test.py
import numpy as np
from environment import ShoverWorldEnv


def smoke_test():
    print("Running Smoke Test...")
    env = ShoverWorldEnv(
        render_mode=None,
        n_rows=6, n_cols=9,
        initial_force=40.0,
        unit_force=10.0,
        seed=42
    )

    obs, info = env.reset()
    total_reward = 0.0
    done = False
    steps = 0

    while not done and steps < 200:
        # Random valid position
        r = np.random.randint(0, env.n_rows)
        c = np.random.randint(0, env.n_cols)

        # Random VALID action ID (1 to 6)
        # 0 is invalid in the new strict spec
        z = np.random.randint(1, 7)

        action = ((r, c), z)

        obs, reward, term, trunc, info = env.step(action)

        total_reward += reward
        steps += 1

        if term or trunc:
            done = True

    print(f"Success! Episode finished in {steps} steps.")
    print(f"Total Reward: {total_reward}")
    print(f"Final Info: {info}")
    env.close()


if __name__ == "__main__":
    smoke_test()
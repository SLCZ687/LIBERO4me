# new_tasks/scripts/block_test/rollout_collect.py

import os
import numpy as np
from libero.libero.envs import OffScreenRenderEnv

from check_success import check_success


OUT_DIR = "./rollouts"
MAX_STEPS = 200


def random_action(robot):
    """
    生成一个合法 action（7 维）
    """
    low, high = robot.action_limits
    return np.random.uniform(low, high)


def rollout_once(bddl_file, rollout_id=0):
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=128,
        camera_widths=128,
    )

    obs = env.reset()
    robot = env.robots[0]

    traj = {
        "actions": [],
        "success": False,
    }

    for t in range(MAX_STEPS):
        action = random_action(robot)
        obs, reward, done, info = env.step(action)

        traj["actions"].append(action)

        if check_success(env):
            traj["success"] = True
            print(f"[✓] success at step {t}")
            break

    env.close()
    return traj


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    bddl = "new_tasks/scripts/block_test/tmp_pddl_files/THREE_BRICK_SCENE_stack_three_brick_cubes.bddl"

    traj = rollout_once(bddl)

    save_path = os.path.join(OUT_DIR, "rollout_000.npz")
    np.savez(save_path, **traj)

    print(f"saved to {save_path}")
    print("success =", traj["success"])


if __name__ == "__main__":
    main()

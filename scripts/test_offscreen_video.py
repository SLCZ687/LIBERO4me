import argparse
import os
import time
import numpy as np
import imageio

from robosuite import load_controller_config

import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs import TASK_MAPPING


def to_uint8(img):
    """Ensure HWC uint8 for video writing."""
    img = np.asarray(img)
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bddl-file", type=str, required=True)
    parser.add_argument("--camera", type=str, default="agentview")
    parser.add_argument("--out", type=str, default="./test_videos")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--controller", type=str, default="OSC_POSE")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--flip", action="store_true", help="Flip frames vertically (fix upside-down).")
    parser.add_argument("--name", type=str, default="", help="Optional video name prefix (no extension).")
    args = parser.parse_args()

    assert os.path.exists(args.bddl_file), f"BDDL not found: {args.bddl_file}"

    problem_info = BDDLUtils.get_problem_info(args.bddl_file)
    problem_name = problem_info["problem_name"]
    language_instruction = problem_info["language_instruction"]
    print("[task]", problem_name)
    print("[lang]", language_instruction)

    controller_config = load_controller_config(default_controller=args.controller)

    config = {
        "robots": ["Panda"],
        "controller_configs": controller_config,
    }
    if "TwoArm" in problem_name:
        # keep as-is from your draft; adjust only if you actually use two-arm tasks
        config["env_configuration"] = "single-arm-opposed"

    # Offscreen-only environment (no GUI window)
    env = TASK_MAPPING[problem_name](
        bddl_file_name=args.bddl_file,
        **config,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera=args.camera,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=args.fps,
    )

    env.seed(args.seed)
    env.reset()

    # Determine action dimension safely
    if hasattr(env, "action_dim"):
        action_dim = env.action_dim
    else:
        action_dim = env.action_space.shape[0]

    os.makedirs(args.out, exist_ok=True)
    base = args.name.strip()
    if base:
        tag = f"{base}_{args.camera}_{time.time():.6f}".replace(".", "_")
    else:
        tag = f"{problem_name}_{args.camera}_{time.time():.6f}".replace(".", "_")

    video_path = os.path.join(args.out, f"{tag}.mp4")

    final_path = video_path
    tmp_path = video_path + ".tmp.mp4"

    writer = imageio.get_writer(
        tmp_path,
        fps=args.fps,
        codec="libx264",
        pixelformat="yuv420p",
    )

    frames = 0
    try:
        for _ in range(args.steps):
            action = np.zeros(action_dim, dtype=np.float32)
            env.step(action)

            frame = env.sim.render(
                camera_name=args.camera,
                width=args.width,
                height=args.height,
                depth=False,
            )
            frame = to_uint8(frame)
            if args.flip:
                frame = np.flipud(frame)

            writer.append_data(frame)
            frames += 1
    finally:
        # make sure both video and env are closed even if error / Ctrl+C
        try:
            writer.close()
        finally:
            env.close()

    if frames > 0:
        os.replace(tmp_path, final_path)
        print("[ok] wrote video:", final_path, "frames:", frames)
    else:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        print("[warn] wrote 0 frames, no video saved")


if __name__ == "__main__":
    main()

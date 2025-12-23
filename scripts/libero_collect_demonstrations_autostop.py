#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
libero_collect_demonstrations_autostop.py

- 用 SpaceMouse / Keyboard 做人类示教采集
- DataCollectionWrapper 记录 state_*.npz
- 每个 demo 结束后打包成 demo.hdf5
- 自定义 success 检测（stack3 宽松版），满足后自动结束当前 demo

依赖：
  pip install h5py opencv-python termcolor
  SpaceMouse 需要 robosuite.devices.SpaceMouse 的依赖已装好
"""

import argparse
import cv2
import datetime
import h5py
import init_path  # noqa: F401  (确保路径/环境初始化)
import json
import numpy as np
import os
import robosuite as suite
import time
from glob import glob
from robosuite import load_controller_config
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper
from robosuite.utils.input_utils import input2action
import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs import *  # TASK_MAPPING, etc.
from termcolor import colored

# 你的宽松成功检测函数（你已经写好了）
from scripts.check_success_stack3 import check_stack3_loose


# ============= Robosuite 1.4.0 OpenCVRenderer: 手动捕获键盘 =============
def manual_keyboard_capture(env, device):
    key = cv2.waitKey(1) & 0xFF
    if key != 255:
        try:
            char_key = chr(key)
            if hasattr(device, "on_press"):
                device.on_press(char_key)
                device.on_release(char_key)
        except ValueError:
            pass


def collect_human_trajectory(env, device, arm, env_configuration, problem_info, args, remove_directory):
    # reset 可能失败，稳一点
    reset_success = False
    while not reset_success:
        try:
            env.reset()
            reset_success = True
        except Exception:
            continue

    # 清掉上一局的计数（check_stack3_loose 可能用 env._stack_hold）
    if hasattr(env, "_stack_hold"):
        env._stack_hold = 0

    env.render()
    device.start_control()
    saving = True
    count = 0

    print("【开始采集】请点击弹出的 OpenCV 窗口开始控制。")

    # 窗口位置/尺寸（不保证所有版本都叫这个名字，但无伤大雅）
    try:
        window_name = "offscreen render"
        position_x, position_y = 100, 100
        size = 640
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(window_name, position_x, position_y)
        cv2.resizeWindow(window_name, size, size)
        print(f"【提示】窗口移动到 ({position_x}, {position_y})，大小 {size}x{size}")
    except Exception as e:
        print(f"窗口调整失败（可忽略）: {e}")

    # 成功稳定帧数（双保险：check_stack3_loose 内部也有 hold；这里再加一层）
    success_hold = 0
    SUCCESS_HOLD_N = args.success_hold

    while True:
        count += 1

        # keyboard 需要手动喂事件
        if args.device == "keyboard":
            manual_keyboard_capture(env, device)

        # 选择 active_robot（单臂基本就是 env.robots[0]）
        active_robot = env.robots[0] if env_configuration != "bimanual" else env.robots[0]

        action, grasp = input2action(
            device=device,
            robot=active_robot,
            active_arm=arm,
            env_configuration=env_configuration,
        )

        # action None 通常表示用户退出/停止
        if action is None:
            print("用户中止，本条 demo 不保存。")
            saving = False
            break

        env.step(action)
        env.render()

        # ====== 自定义成功检测（宽松版）======
        ok = check_stack3_loose(
            env,
            hold_steps=args.loose_hold,   # 这个是 check_stack3_loose 自己的连续帧要求
            xy_tol=args.xy_tol,
            min_z_gap=args.min_z_gap,
        )
        if ok:
            success_hold += 1
            if success_hold >= SUCCESS_HOLD_N:
                print(colored("✅ TASK SUCCESS! Ending this demo.", "green", attrs=["bold"]))
                break
        else:
            success_hold = 0

        if args.debug and count % 20 == 0:
            try:
                print("debug: env._check_success() =", env._check_success())
            except Exception:
                pass

    print("steps in this demo =", count)

    # 不中断保存逻辑：不中断才保存
    if not saving:
        try:
            remove_directory.append(env.ep_directory.split("/")[-1])
        except Exception:
            pass

    env.close()
    return saving


def gather_demonstrations_as_hdf5(tmp_directory, out_dir, env_info, args, problem_info, remove_directory):
    """
    把 DataCollectionWrapper 产生的 state_*.npz 打包成 demo.hdf5
    注意：每次调用会覆盖 out_dir/demo.hdf5（与同学脚本一致）
    """
    os.makedirs(out_dir, exist_ok=True)
    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    f = h5py.File(hdf5_path, "w")
    grp = f.create_group("data")

    num_eps = 0
    env_name = None

    if not os.path.isdir(tmp_directory):
        print("WARNING: tmp_directory not found:", tmp_directory)
        f.close()
        return

    for ep_directory in sorted(os.listdir(tmp_directory)):
        if ep_directory in remove_directory:
            continue

        state_paths = os.path.join(tmp_directory, ep_directory, "state_*.npz")
        states, actions = [], []

        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])
            states.extend(dic["states"])
            for ai in dic["action_infos"]:
                actions.append(ai["actions"])

        if len(states) == 0:
            continue

        # robosuite 的 states 比 actions 多 1
        del states[-1]

        num_eps += 1
        ep_data_grp = grp.create_group(f"demo_{num_eps}")

        xml_path = os.path.join(tmp_directory, ep_directory, "model.xml")
        if os.path.exists(xml_path):
            with open(xml_path, "r") as fx:
                ep_data_grp.attrs["model_file"] = fx.read()

        ep_data_grp.create_dataset("states", data=np.array(states))
        ep_data_grp.create_dataset("actions", data=np.array(actions))

    now = datetime.datetime.now()
    grp.attrs["date"] = f"{now.month}-{now.day}-{now.year}"
    grp.attrs["time"] = f"{now.hour}:{now.minute}:{now.second}"
    grp.attrs["repository_version"] = suite.__version__
    grp.attrs["env"] = env_name if env_name is not None else ""
    grp.attrs["env_info"] = env_info
    grp.attrs["problem_info"] = json.dumps(problem_info)
    grp.attrs["bddl_file_name"] = args.bddl_file

    try:
        with open(args.bddl_file, "r", encoding="utf-8") as bf:
            grp.attrs["bddl_file_content"] = bf.read()
    except Exception:
        grp.attrs["bddl_file_content"] = ""

    f.close()
    print(colored(f"[saved] {hdf5_path} (num demos packed = {num_eps})", "cyan"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, default="demonstration_data")
    parser.add_argument("--robots", nargs="+", type=str, default=["Panda"])
    parser.add_argument("--config", type=str, default="single-arm-opposed")
    parser.add_argument("--arm", type=str, default="right")
    parser.add_argument("--camera", type=str, default="agentview")
    parser.add_argument("--controller", type=str, default="OSC_POSE")
    parser.add_argument("--device", type=str, default="spacemouse", choices=["spacemouse", "keyboard"])
    parser.add_argument("--pos-sensitivity", type=float, default=1.5)
    parser.add_argument("--rot-sensitivity", type=float, default=1.5)
    parser.add_argument("--num-demonstration", type=int, default=50)
    parser.add_argument("--bddl-file", type=str, required=True)
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--vendor-id", type=int, default=9583)
    parser.add_argument("--product-id", type=int, default=50746)

    # success 判定参数
    parser.add_argument("--xy-tol", type=float, default=0.12)
    parser.add_argument("--min-z-gap", type=float, default=0.02)
    parser.add_argument("--loose-hold", type=int, default=15)     # check_stack3_loose 内部 hold_steps
    parser.add_argument("--success-hold", type=int, default=10)   # 外层再 hold 一次
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    assert os.path.exists(args.bddl_file), f"bddl not found: {args.bddl_file}"

    controller_config = load_controller_config(default_controller=args.controller)
    config = {
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    problem_info = BDDLUtils.get_problem_info(args.bddl_file)
    problem_name = problem_info["problem_name"]
    domain_name = problem_info["domain_name"]
    language_instruction = problem_info["language_instruction"]

    print("Goal:", colored(language_instruction, "red", attrs=["bold"]))
    print(colored("Hit ENTER to proceed...", "green", attrs=["reverse"]))
    input()

    if "TwoArm" in problem_name:
        config["env_configuration"] = args.config

    # ===== 创建环境（GUI / 人机交互）=====
    env = TASK_MAPPING[problem_name](
        bddl_file_name=args.bddl_file,
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera=args.camera,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )

    env = VisualizationWrapper(env)

    # DataCollectionWrapper 的临时目录（每条 demo 会生成 ep_xxx）
    tmp_directory = "test/demonstration_data/tmp/{}_ln_{}/{}".format(
        problem_name,
        language_instruction.replace(" ", "_").strip('""'),
        str(time.time()).replace(".", "_"),
    )
    env = DataCollectionWrapper(env, tmp_directory)

    env.reset()
    env.render()

    # ===== 选择设备 =====
    if args.device == "keyboard":
        from robosuite.devices import Keyboard
        device = Keyboard(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
        print("Keyboard initialized. (OpenCVRenderer needs manual capture)")
    else:
        from robosuite.devices import SpaceMouse
        device = SpaceMouse(
            args.vendor_id,
            args.product_id,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
        )
        print("SpaceMouse initialized.")

    # ===== 输出目录（真正的 demo.hdf5 放这里）=====
    t1, t2 = str(time.time()).split(".")
    out_dir = os.path.join(
        args.directory,
        f"{domain_name}_ln_{problem_name}_{t1}_{t2}_" + language_instruction.replace(" ", "_").strip('""'),
    )
    os.makedirs(out_dir, exist_ok=True)
    print("Output dir:", colored(out_dir, "yellow"))

    env_info = json.dumps(config)
    remove_directory = []

    i = 0
    while i < args.num_demonstration:
        print(colored(f"\n===== Collect Demo {i} / {args.num_demonstration} =====", "magenta"))
        saving = collect_human_trajectory(
            env=env,
            device=device,
            arm=args.arm,
            env_configuration=args.config,
            problem_info=problem_info,
            args=args,
            remove_directory=remove_directory,
        )
        if saving:
            gather_demonstrations_as_hdf5(
                tmp_directory=tmp_directory,
                out_dir=out_dir,
                env_info=env_info,
                args=args,
                problem_info=problem_info,
                remove_directory=remove_directory,
            )
            i += 1

    print(colored("All done.", "green", attrs=["bold"]))


if __name__ == "__main__":
    main()

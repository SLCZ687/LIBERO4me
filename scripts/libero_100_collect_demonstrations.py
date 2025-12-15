import argparse
import cv2
import datetime
import h5py
import init_path
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
from libero.libero.envs import *
from termcolor import colored

# Robosuite 1.4.0 的 OpenCVRenderer 没有回调函数
# 我们需要在主循环中手动捕获按键
def manual_keyboard_capture(env, device):
    # 捕获 OpenCV 窗口的按键 (等待 1ms)
    # 注意：必须有窗口焦点
    key = cv2.waitKey(1) & 0xFF
    if key != 255: # 255 表示没有按键
        # 模拟按下和松开
        try:
            # 将 ASCII 码转换为字符
            char_key = chr(key)
            # 只有当设备是 Keyboard 时才处理
            if hasattr(device, "on_press"):
                device.on_press(char_key)
                # 简单的逻辑：按下一瞬间后立即释放，或者你可以根据逻辑调整
                # 键盘控制在 Robot Learning 中通常比较生硬
                device.on_release(char_key)
        except ValueError:
            pass

def collect_human_trajectory(
    env, device, arm, env_configuration, problem_info, remove_directory=[]
):
    reset_success = False
    while not reset_success:
        try:
            env.reset()
            reset_success = True
        except:
            continue

    env.render()
    task_completion_hold_count = -1
    device.start_control()
    saving = True
    count = 0

    print("【开始采集】请点击弹出的 OpenCV 窗口，使用 W/A/S/D/Space 控制...")

    # =================【窗口位置】=================
    # Robosuite 的 VisualizationWrapper 默认窗口名叫 "Visualization"
    try:
        window_name = "offscreen render"
        # 确保窗口已被创建（有些版本需要先 namedWindow）
        position_x = 100
        position_y = 100
        size = 512
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) 
        cv2.moveWindow(window_name, position_x, position_y)
        # 调整大小
        cv2.resizeWindow(window_name, size, size)
        print(f"【提示】已将窗口移动到左上角 ({position_x}, {position_y})")
    except Exception as e:
        print(f"窗口调整失败: {e}")
    # ======================================================

    while True:
        count += 1
        
        # 【关键修改】手动捕获键盘输入
        # 因为 Robosuite 1.4.0 移除了自动回调
        if args.device == "keyboard":
            manual_keyboard_capture(env, device)

        active_robot = (
            env.robots[0]
            if env_configuration == "bimanual"
            else env.robots[arm == "left"]
        )

        action, grasp = input2action(
            device=device,
            robot=active_robot,
            active_arm=arm,
            env_configuration=env_configuration,
        )

        if action is None:
            print("Break")
            saving = False
            break

        env.step(action)
        env.render()

        if task_completion_hold_count == 0:
            break

        if env._check_success():
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1
            else:
                task_completion_hold_count = 10
        else:
            task_completion_hold_count = -1

    print(count)
    if not saving:
        remove_directory.append(env.ep_directory.split("/")[-1])
    env.close()
    return saving


def gather_demonstrations_as_hdf5(directory, out_dir, env_info, args, remove_directory=[]):
    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    f = h5py.File(hdf5_path, "w")
    grp = f.create_group("data")
    num_eps = 0
    env_name = None

    for ep_directory in os.listdir(directory):
        if ep_directory in remove_directory:
            continue
        state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        states = []
        actions = []

        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])
            states.extend(dic["states"])
            for ai in dic["action_infos"]:
                actions.append(ai["actions"])

        if len(states) == 0: continue
        del states[-1]
        
        num_eps += 1
        ep_data_grp = grp.create_group("demo_{}".format(num_eps))
        xml_path = os.path.join(directory, ep_directory, "model.xml")
        with open(xml_path, "r") as f:
            xml_str = f.read()
        ep_data_grp.attrs["model_file"] = xml_str
        ep_data_grp.create_dataset("states", data=np.array(states))
        ep_data_grp.create_dataset("actions", data=np.array(actions))

    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["repository_version"] = suite.__version__
    grp.attrs["env"] = env_name
    grp.attrs["env_info"] = env_info
    grp.attrs["problem_info"] = json.dumps(problem_info)
    grp.attrs["bddl_file_name"] = args.bddl_file
    grp.attrs["bddl_file_content"] = str(open(args.bddl_file, "r", encoding="utf-8"))
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, default="demonstration_data")
    parser.add_argument("--robots", nargs="+", type=list, default=["Panda"])
    parser.add_argument("--config", type=str, default="single-arm-opposed")
    parser.add_argument("--arm", type=str, default="right")
    parser.add_argument("--camera", type=str, default="agentview")
    parser.add_argument("--controller", type=str, default="OSC_POSE")
    parser.add_argument("--device", type=str, default="spacemouse")
    parser.add_argument("--pos-sensitivity", type=float, default=1.5)
    parser.add_argument("--rot-sensitivity", type=float, default=1.5)
    parser.add_argument("--num-demonstration", type=int, default=50)
    parser.add_argument("--bddl-file", type=str, default=None)
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--vendor-id", type=int, default=9583)
    parser.add_argument("--product-id", type=int, default=50746)
    args = parser.parse_args()

    controller_config = load_controller_config(default_controller=args.controller)
    config = {
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    assert os.path.exists(args.bddl_file)
    problem_info = BDDLUtils.get_problem_info(args.bddl_file)
    problem_name = problem_info["problem_name"]
    domain_name = problem_info["domain_name"]
    language_instruction = problem_info["language_instruction"]
    
    print("Goal: ", colored(language_instruction, "red", attrs=["bold"]))
    print(colored("Hit any key to proceed...", "green", attrs=["reverse", "blink"]))
    input()

    if "TwoArm" in problem_name:
        config["env_configuration"] = args.config

    # 创建环境 (Robosuite 1.4.0 默认使用 OpenCVRenderer)
    env = TASK_MAPPING[problem_name](
        bddl_file_name=args.bddl_file,
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera=args.camera, # 使用指定的相机视角
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )

    env = VisualizationWrapper(env)
    env_info = json.dumps(config)
    tmp_directory = "test/demonstration_data/tmp/{}_ln_{}/{}".format(
        problem_name,
        language_instruction.replace(" ", "_").strip('""'),
        str(time.time()).replace(".", "_"),
    )
    env = DataCollectionWrapper(env, tmp_directory)

    env.reset()
    env.render() # 初始化窗口

    if args.device == "keyboard":
        from robosuite.devices import Keyboard
        device = Keyboard(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
        # 【修改】Robosuite 1.4.0 不支持回调，这里什么都不做
        # 我们将在主循环中手动调用 manual_keyboard_capture
        print("Keyboard initialized. callbacks disabled for OpenCVRenderer.")
        
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse
        device = SpaceMouse(args.vendor_id, args.product_id, pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    else:
        raise Exception("Invalid device choice")

    t1, t2 = str(time.time()).split(".")
    new_dir = os.path.join(
        args.directory,
        f"{domain_name}_ln_{problem_name}_{t1}_{t2}_" + language_instruction.replace(" ", "_").strip('""'),
    )
    os.makedirs(new_dir)

    remove_directory = []
    i = 0
    while i < args.num_demonstration:
        print(f"Collection Demo {i}...")
        saving = collect_human_trajectory(
            env, device, args.arm, args.config, problem_info, remove_directory
        )
        if saving:
            gather_demonstrations_as_hdf5(tmp_directory, new_dir, env_info, args, remove_directory)
            i += 1
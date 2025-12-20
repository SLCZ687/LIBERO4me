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
import datetime

WINDOW_SIZE = 768

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
                # 简单的逻辑：按下一瞬间后立即释放
                device.on_release(char_key)
        except ValueError:
            pass

def collect_human_trajectory(
    env, device, arm, env_configuration, problem_info, args, remove_directory=[]
):
    """
    修改后的采集循环：
    不再使用 env.render()，而是手动获取两个相机的图像进行拼接显示。
    """
    reset_success = False
    while not reset_success:
        try:
            env.reset()
            reset_success = True
        except:
            continue

    task_completion_hold_count = -1
    device.start_control()
    saving = True
    count = 0

    print("【开始采集】请点击弹出的 OpenCV 窗口，使用 W/A/S/D/Space 控制...")
    print(f"当前视角: 左[{args.camera}] | 右[{args.camera2}]")

    # =================【窗口设置】=================
    window_name = "Dual View Data Collection"
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) 
        # 两个 512x512 的图像拼接，宽 1024，高 512
        cv2.resizeWindow(window_name, WINDOW_SIZE*2, WINDOW_SIZE)
        cv2.moveWindow(window_name, 100, 100)
    except Exception as e:
        print(f"窗口调整失败: {e}")
    # ==============================================

    while True:
        count += 1
        
        # =================【获取双视角图像】=================
        # 1. 获取主视角图像 (例如 agentview)
        # env.sim.render 返回的是 RGB 且上下颠倒的数组，需要处理
        obs_img_1 = env.sim.render(camera_name=args.camera, width=512, height=512)
        obs_img_1 = cv2.cvtColor(np.flipud(obs_img_1), cv2.COLOR_RGB2BGR)

        # 2. 获取第二视角图像 (例如 robot0_robotview 或 sideview)
        obs_img_2 = env.sim.render(camera_name=args.camera2, width=512, height=512)
        obs_img_2 = cv2.cvtColor(np.flipud(obs_img_2), cv2.COLOR_RGB2BGR)

        # 3. 水平拼接 (Left: Main, Right: Aux)
        display_img = np.hstack([obs_img_1, obs_img_2])

        # 4. 在图像上叠加文字信息 (可选)
        cv2.putText(display_img, f"Main: {args.camera}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display_img, f"Aux: {args.camera2}", (512 + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 5. 显示
        cv2.imshow(window_name, display_img)
        # ====================================================

        # =================【输入处理】=================
        # 如果是键盘控制，需要捕获按键并传给 device
        if args.device == "keyboard":
            manual_keyboard_capture(env, device)
        else:
            # 如果是 SpaceMouse，也必须调用 waitKey 来刷新 OpenCV 窗口
            cv2.waitKey(1)
        # ==============================================

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
        # env.render() # 【已移除】我们自己处理了显示

        if task_completion_hold_count == 0:
            break

        if env._check_success():
            # 成功时在屏幕上画绿框提示
            cv2.rectangle(display_img, (0, 0), (1024, 512), (0, 255, 0), 10)
            cv2.imshow(window_name, display_img)
            cv2.waitKey(1)

            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1
            else:
                task_completion_hold_count = 10
        else:
            task_completion_hold_count = -1

    print(f"Trajectory length: {count}")
    
    cv2.destroyWindow(window_name) # 关闭窗口
    
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
    if args.bddl_file:
         with open(args.bddl_file, "r", encoding="utf-8") as f:
            grp.attrs["bddl_file_content"] = f.read()
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, default="demonstration_data")
    parser.add_argument("--robots", nargs="+", type=list, default=["Panda"])
    parser.add_argument("--config", type=str, default="single-arm-opposed")
    parser.add_argument("--arm", type=str, default="right")
    
    # 【新增】相机参数
    parser.add_argument("--camera", type=str, default="agentview", help="主视角")
    # 默认使用 robot0_robotview (第三人称)，如果不喜欢可以换成 sideview
    # parser.add_argument("--camera2", type=str, default="robot0_robotview", help="辅助视角，用于提供空间关系")
    # Available "camera" names = ('frontview', 'birdview', 'agentview', 
    # 'sideview', 'galleryview', 'paperview', 'robot0_robotview', 'robot0_eye_in_hand').
    parser.add_argument("--camera2", type=str, default="sideview", help="辅助视角，用于提供空间关系")
    
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

    # 【关键修改】配置环境渲染模式
    # has_renderer=False: 关闭 Mujoco 原生 viewer
    # has_offscreen_renderer=True: 开启后台渲染，以便我们手动提取图像
    env = TASK_MAPPING[problem_name](
        bddl_file_name=args.bddl_file,
        **config,
        has_renderer=False, 
        has_offscreen_renderer=True,
        render_camera=args.camera, 
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )

    env = VisualizationWrapper(env)
    env_info = json.dumps(config)
    tmp_directory = "data/{}_ln_{}/{}".format(
        problem_name,
        language_instruction.replace(" ", "_").strip('""'),
        str(time.time()).replace(".", "_"),
    )
    env = DataCollectionWrapper(env, tmp_directory)

    env.reset()
    # env.render() # 已移除，我们在主循环中处理

    if args.device == "keyboard":
        from robosuite.devices import Keyboard
        device = Keyboard(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
        print("Keyboard initialized.")
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
        # 将 args 传入函数，以便读取 camera2 配置
        saving = collect_human_trajectory(
            env, device, args.arm, args.config, problem_info, args, remove_directory
        )
        if saving:
            gather_demonstrations_as_hdf5(tmp_directory, new_dir, env_info, args, remove_directory)
            i += 1
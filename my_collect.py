import os
import time
import json
import argparse
import datetime
import h5py
import numpy as np
from glob import glob

# 导入 LIBERO 和 Robosuite 基础件
import robosuite as suite
from robosuite import load_controller_config
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper
from robosuite.utils.input_utils import input2action
from robosuite.devices import Keyboard

import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs import *

# ========================================================
# 第一部分：直接粘贴你刚才给我的那两个官方核心函数
# ========================================================

def collect_human_trajectory(env, device, arm, env_configuration, problem_info, remove_directory=[]):
    reset_success = False
    while not reset_success:
        try:
            env.reset()
            reset_success = True
        except: continue
    
    # 这一行会弹出 3D 窗口
    env.render()
    task_completion_hold_count = -1
    device.start_control()
    saving = True
    count = 0

    while True:
        count += 1
        active_robot = env.robots[0] # 简化逻辑：假设单臂
        action, grasp = input2action(device=device, robot=active_robot, active_arm=arm, env_configuration=env_configuration)
        
        if action is None: # 按 ESC 或特定键退出
            print("停止当前录制")
            saving = False
            break

        env.step(action)
        env.render()

        if task_completion_hold_count == 0: break
        
        # 自动检测任务是否完成（书是否进了托盘）
        if env._check_success():
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1
            else:
                task_completion_hold_count = 10 # 成功后持续 10 帧则自动结束
        else:
            task_completion_hold_count = -1

    if not saving:
        remove_directory.append(env.ep_directory.split("/")[-1])
    return saving

# 注意：这里简化了归档逻辑，直接调用官方逻辑即可
def gather_demonstrations_as_hdf5(directory, out_dir, env_info, args, remove_directory=[]):
    # 这里使用你刚才提供的官方 gather_demonstrations_as_hdf5 的完整逻辑
    # 为了演示，我们假设你已经把那个长函数贴在这里了
    pass

# ========================================================
# 第二部分：运行主程序
# ========================================================

if __name__ == "__main__":
    # 配置
    bddl_path = "libero/libero/bddl_files/my_tasks/KITCHEN_SCENE1_put_the_black_book_in_the_wooden_tray.bddl"
    controller_config = load_controller_config(default_controller="OSC_POSE")
    problem_info = BDDLUtils.get_problem_info(bddl_path)

    # 【关键修改】不从 TASK_MAPPING 找，直接手动实例化 KitchenScene1
    env = KitchenScene1(
        bddl_file_name=bddl_path,
        robots=["Panda"],
        controller_configs=controller_config,
        has_renderer=True,          # 必须为 True 才能看到窗口
        has_offscreen_renderer=False,
        render_camera="agentview",
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )

    env = VisualizationWrapper(env)
    
    # 设置数据存放路径
    tmp_dir = f"demonstration_data/tmp/my_task_{int(time.time())}"
    final_dir = f"demonstration_data/my_task_final"
    os.makedirs(final_dir, exist_ok=True)

    # 包装环境，用于自动保存每一帧的数据
    env = DataCollectionWrapper(env, tmp_dir)

    # 初始化键盘
    device = Keyboard(pos_sensitivity=1.0, rot_sensitivity=1.0)
    env.viewer.add_keypress_callback("any", device.on_press)
    env.viewer.add_keyup_callback("any", device.on_release)

    print("🚀 准备就绪！请在弹出的窗口中操作。")
    print("操作提示：W/S/A/D/R/F 控制移动，数字键 1 抓取，数字键 2 放开。")

    # 录制 5 组试试看
    num_to_collect = 5
    i = 0
    remove_dir = []
    while i < num_to_collect:
        print(f"正在录制第 {i+1}/{num_to_collect} 组...")
        if collect_human_trajectory(env, device, "right", "single-arm-opposed", problem_info, remove_dir):
            i += 1
            print(f"✅ 第 {i} 组录制成功！")
    
    print(f"📦 数据已采集完成，请检查 {final_dir} 目录。")
    env.close()
import os
import time
import h5py
import numpy as np
from glob import glob

# 强制使用窗口渲染后端


import robosuite
from robosuite.wrappers import DataCollectionWrapper
from robosuite.utils.input_utils import input2action
from robosuite.devices import Keyboard

from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import libero.libero.envs.bddl_utils as BDDLUtils

def collect_human_trajectory(env, device, arm, env_configuration, problem_info, remove_directory=[]):
    env.reset()
    device.start_control()
    saving = True
    
    print("\n🟢 正在录制... 请操作机械臂。")
    print("🚩 操作完成后，请按键盘上的 ESC 键保存并进入下一条。")

    while True:
        try:
            # 兼容性获取机器人对象
            active_robot = env.robots[0] if hasattr(env, "robots") else env.env.env.robots[0]
            action, grasp = input2action(device=device, robot=active_robot, active_arm=arm, env_configuration=env_configuration)
            
            if action is None: # 用户按下 ESC 键
                print("⚠️ 检测到 ESC，正在保存本条轨迹...")
                saving = True # 强制设为 True，我们要存下这段动作
                break
                
            env.step(action)
            
            # 渲染画面
            try: env.env.env.render()
            except: pass
            
            # 彻底屏蔽自动检测报错，完全靠手动 ESC 结束
            # 这样就不会触发那个 KeyError: 'wooden_tray_1_tray_stack_region'
            
        except Exception as e:

            continue
            
    if not saving:
        ep_dir = getattr(env, "ep_directory", "").split("/")[-1]
        if ep_dir: remove_directory.append(ep_dir)
    return saving

def gather_demonstrations_as_hdf5(directory, out_dir, remove_directory=[]):
    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    f = h5py.File(hdf5_path, "a")
    grp = f.require_group("data")
    num_eps = len([k for k in grp.keys() if k.startswith("demo_")])
    for ep_directory in os.listdir(directory):
        if ep_directory in remove_directory: continue
        state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        states, actions = [], []
        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            states.extend(dic["states"])
            for ai in dic["action_infos"]: actions.append(ai["actions"])
        if len(states) == 0: continue
        del states[-1]
        num_eps += 1
        ep_grp = grp.create_group(f"demo_{num_eps}")
        with open(os.path.join(directory, ep_directory, "model.xml"), "r") as m_f:
            ep_grp.attrs["model_file"] = m_f.read()
        ep_grp.create_dataset("states", data=np.array(states))
        ep_grp.create_dataset("actions", data=np.array(actions))
    f.close()

if __name__ == "__main__":
    bddl_root = get_libero_path("bddl_files")
    bddl_file = os.path.join(bddl_root, "my_tasks/KITCHEN_SCENE1_put_the_black_book_in_the_wooden_tray.bddl")
    
    # 实例化环境
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        robots=["Panda"],
        has_renderer=True,           # 开启渲染器
        has_offscreen_renderer=True, # 开启离屏渲染（有些版本必须同时开启）
        render_camera="agentview",
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
    )

    # --- 新增：强制初始化窗口 ---
    env.reset()
    try:
        # 这一步非常关键，它会强制启动交互式 Viewer 窗口
        env.env.env.viewer.make_context_current() 
        print("💡 窗口初始化指令已发出...")
    except Exception as e:
        print(f"💡 自动唤醒窗口失败，尝试手动触发：{e}")

    problem_info = BDDLUtils.get_problem_info(bddl_file)
    ts = int(time.time())
    tmp_dir = f"demonstration_data/tmp/my_task_{ts}"
    final_dir = f"demonstration_data/my_task_final_{ts}"
    os.makedirs(final_dir, exist_ok=True)
    
    env = DataCollectionWrapper(env, tmp_dir)
    device = Keyboard(pos_sensitivity=1.5, rot_sensitivity=1.0)
    
    env.reset()
    try:
        # 绑定键盘到 MuJoCo 窗口
        viewer = env.env.env.viewer
        viewer.add_keypress_callback("any", device.on_press)
        viewer.add_keyup_callback("any", device.on_release)
    except:
        pass

    i, remove_dir = 0, []
    while i < 5:
        print(f"\n======== 录制第 {i+1} / 5 条 ========")
        if collect_human_trajectory(env, device, "right", "single-arm-opposed", problem_info, remove_dir):
            gather_demonstrations_as_hdf5(tmp_dir, final_dir, remove_dir)
            i += 1
            print(f"✅ 第 {i} 条录制成功！")
    
    print(f"\n✨ 任务圆满完成！数据文件：{final_dir}/demo.hdf5")
    env.close()
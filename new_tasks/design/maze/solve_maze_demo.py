import robosuite as suite
import numpy as np
import imageio
import os
import robosuite.utils.transform_utils as T
from datetime import datetime

# 导入你的任务
import my_maze_task 

# 设置环境变量
os.environ["MUJOCO_GL"] = "egl"

def solve_maze():
    print(">>> 正在初始化迷宫环境 (OSC_POSE 控制模式)...")

    # 1. 加载 OSC_POSE 控制器配置
    controller_config = suite.load_controller_config(default_controller="OSC_POSE")

    # 2. 创建环境
    env = suite.make(
        env_name="MazeNavigationTask",
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names="frontview", 
        render_camera="frontview",
        camera_heights=512,
        camera_widths=512,
        control_freq=20,
        horizon=600, 
    )

    obs = env.reset()
    print(">>> 环境创建成功！开始执行自动推球策略...")

    # 3. 准备视频保存
    result_dir = "result"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join(result_dir, f"solve_maze_{timestamp}.mp4")
    video_frames = []

    # 获取正确的路径点
    MAZE_BASE_HEIGHT = 0.82
    # 为了防止手擦到地面，目标高度设为比地板高一点点
    TARGET_Z = MAZE_BASE_HEIGHT + 0.03 
    
    waypoints = env.waypoints 

    # 4. 仿真循环
    print(f">>> 目标路径点数量: {len(waypoints)}")
    
    kp = 6.0 # 比例增益

    for i in range(env.horizon):
        # --- A. 感知 ---
        ball_pos = obs["ball_pos"]
        hand_pos = obs["robot0_eef_pos"]

        # --- B. 策略 (简单的状态机) ---
        # 1. 找到离球最近的路径点索引
        dists = np.linalg.norm(waypoints[:, :2] - ball_pos[:2], axis=1)
        current_idx = np.argmin(dists)

        # 2. 确定下一个目标点 (Lookahead)
        target_idx = min(current_idx + 2, len(waypoints) - 1)
        target_wp = waypoints[target_idx]
        
        # 目标向量：从球指向下一个路径点
        direction_vec = target_wp[:2] - ball_pos[:2]
        dist_to_target = np.linalg.norm(direction_vec)
        
        if dist_to_target < 0.001:
            direction_vec = np.array([1.0, 0.0])
        else:
            direction_vec = direction_vec / dist_to_target

        # 3. 计算机械臂的目标位置 (推球点)
        push_offset = -0.04 
        target_hand_xy = ball_pos[:2] + direction_vec * push_offset
        
        # 4. 动作生成 (P-Control)
        error_xy = target_hand_xy - hand_pos[:2]
        error_z = TARGET_Z - hand_pos[2]
        
        action_xyz = np.array([error_xy[0], error_xy[1], error_z]) * kp
        action_xyz = np.clip(action_xyz, -1.0, 1.0)
        
        # [关键修正] 姿态控制 + 夹爪控制
        # action 格式: [x, y, z, ax, ay, az, gripper]
        # 前3个是位置，中间3个是姿态(0表示不转)，最后1个是夹爪(-1表示闭合)
        action = np.concatenate([action_xyz, [0, 0, 0], [-1]]) 

        # --- C. 执行 ---
        obs, reward, done, info = env.step(action)
        
        # 打印状态
        if i % 10 == 0:
            progress = env.get_progress()
            print(f"Step {i:03d}: 进度 = {progress:.2%} | 球位置: {ball_pos[:2]}")

        # --- D. 录制 ---
        frame = np.flip(obs["frontview_image"], 0)
        video_frames.append(frame)

    # 5. 保存视频
    print(f">>> 仿真结束，正在保存视频到 {video_path} ...")
    imageio.mimsave(video_path, video_frames, fps=20)
    print(">>> 完成！")
    
    env.close()

if __name__ == "__main__":
    solve_maze()
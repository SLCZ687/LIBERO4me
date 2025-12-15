import robosuite as suite
import numpy as np
import cv2
import os
import imageio
import zmq  # 引入通信库

# 导入你的任务
import my_maze_task 

# === 配置 ===
SERVER_IP = "localhost"
PORT = 5555
os.environ["MUJOCO_GL"] = "egl"

def main():
    # 1. 初始化 ZeroMQ 客户端
    context = zmq.Context()
    socket = context.socket(zmq.REQ) # REQ (Request) 模式
    print(f">>> [身体] 正在连接大脑 {SERVER_IP}:{PORT}...")
    socket.connect(f"tcp://{SERVER_IP}:{PORT}")

    # 2. 创建环境
    print(">>> [身体] 正在初始化迷宫环境...")
    controller_config = suite.load_controller_config(default_controller="OSC_POSE")
    
    env = suite.make(
        env_name="MazeNavigationTask",
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        # 必须提供模型需要的两个视角
        camera_names=["agentview", "robot0_eye_in_hand"], 
        camera_heights=128,
        camera_widths=128,
        control_freq=20,
        horizon=400,
    )

    obs = env.reset()
    print(">>> [身体] 环境就绪，开始循环...")

    video_frames = []

    # 3. 仿真循环
    for i in range(200):
        # --- A. 获取图像 ---
        img_agent = obs["agentview_image"]
        img_wrist = obs["robot0_eye_in_hand_image"]

        # --- B. 发送给大脑 ---
        # 构造数据包
        data_packet = {
            "agentview": img_agent,
            "wrist": img_wrist
        }
        socket.send_pyobj(data_packet) # 发送

        # --- C. 等待接收动作 (这里会暂停，直到大脑算完) ---
        action = socket.recv_pyobj() # 接收

        # --- D. 执行动作 ---
        obs, reward, done, info = env.step(action)

        # 打印进度
        if i % 10 == 0:
            progress = env.get_progress()
            print(f"Step {i:03d}: 进度 = {progress:.2%}")

        # 保存视频帧 (用 agentview 存大一点的图好看，但这里只有128x128)
        # 如果想要高清视频，可以在 make env 时多加一个 frontview camera 用于渲染，不发给模型
        frame = obs["agentview_image"].astype(np.uint8)
        video_frames.append(np.flip(frame, 0))

    # 保存视频
    print(">>> [身体] 正在保存视频...")
    imageio.mimsave("result/distributed_inference.mp4", video_frames, fps=20)
    env.close()
    print(">>> [身体] 完成。")

if __name__ == "__main__":
    main()
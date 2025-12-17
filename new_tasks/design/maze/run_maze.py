import robosuite as suite
import numpy as np
import os
import imageio  # 核心库：用于保存视频
from datetime import datetime # 核心库：用于获取当前时间
import my_maze_task 

# 设置环境变量，防止 EGL 报错
os.environ["MUJOCO_GL"] = "egl"
VIEW = "frontview"
# VIEW = "agentview"
# VIEW = "birdview"

print("正在初始化迷宫环境...")

# 创建环境
env = suite.make(
    env_name="MazeNavigationTask",
    robots="Panda",
    has_renderer=False,          
    has_offscreen_renderer=True, # 开启后台渲染
    use_camera_obs=True,
    camera_names=VIEW,
    render_camera=VIEW,
    camera_heights=512,
    camera_widths=512,
    control_freq=20,             # 控制频率，也是视频的帧率
    horizon=200,                 
)

env.reset()
print("环境创建成功！开始仿真...")

# --- 1. 准备保存路径和文件名 ---
result_dir = "result"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# 获取当前时间，格式如：2023-10-27_10-30-05
# timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# video_path = os.path.join(result_dir, f"{timestamp}.mp4")
video_path = os.path.join(result_dir, f"{VIEW}_video.mp4")

# 用于存储每一帧画面的列表
video_frames = []

# --- 2. 循环运行 ---
# 建议运行步数多一点，50步对于视频来说只有2.5秒，这里改为 100 步演示
run_steps = 20

for i in range(run_steps):
    # 生成随机动作
    action = np.random.randn(env.robots[0].dof) * 0.5
    
    obs, reward, done, info = env.step(action)
    
    # 获取进度并打印
    progress = env.get_progress()
    print(f"Step {i:03d}: 进度 = {progress:.2%}")
    
    # --- 3. 获取图像并存入列表 ---
    # 获取图像
    img = obs[VIEW + "_image"]
    
    # Robosuite 渲染出来的图像是倒着的，需要垂直翻转
    img = np.flip(img, 0)
    
    # 注意：
    # Robosuite 返回的是 RGB 格式。
    # imageio 保存视频也需要 RGB 格式。
    # OpenCV 保存图片需要 BGR 格式。
    # 因为我们要存视频，所以这里 **不需要** cv2.cvtColor 转颜色
    
    # 将 uint8 格式的图片放入列表
    video_frames.append(img)

# --- 4. 保存视频 ---
print(f"仿真结束。正在将 {len(video_frames)} 帧画面保存为视频...")

# fps=20 对应上面的 control_freq=20，保证视频播放速度和仿真速度一致
try:
    imageio.mimsave(video_path, video_frames, fps=20)
    print(f"视频保存成功！路径: {video_path}")
except Exception as e:
    print(f"视频保存失败: {e}")

env.close()
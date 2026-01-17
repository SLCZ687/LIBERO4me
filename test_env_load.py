import os
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv
from libero.libero.utils.time_utils import Timer
from libero.libero.benchmark import get_benchmark
import numpy as np
import cv2 # 用于显示画面

def test_load_door():
    # 1. 指定你的 BDDL 文件名字（不带后缀）
    task_name = "KITCHEN_SCENE1_test_my_door"
    
    # 2. 获取 benchmark 实例
    # LIBERO 会在初始化时扫描 bddl_files/my_tasks 下的所有文件
    benchmark = get_benchmark("libero_10") 
    
    # 3. 构造环境
    # 注意：bddl_file_name 需要指定相对路径
    env_args = {
        "bddl_file_name": os.path.join("libero/libero/envs/bddl_files/my_tasks", f"{task_name}.bddl"),
        "camera_height": 512,
        "camera_width": 512,
        "render_gpu_device_id": 0,
    }

    print(f"正在尝试加载任务: {task_name}...")
    
    # 创建仿真环境
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    env.reset()

    # 4. 循环渲染画面，让你亲眼看到物体
    print("加载成功！正在打开预览窗口（按 'q' 键退出）...")
    for i in range(100):
        # 获取机械臂视角的图像
        obs = env._get_observations()
        img = obs["agentview_image"]
        
        # 转换颜色空间（RGB -> BGR）用于 OpenCV 显示
        img_bgr = cv2.cvtColor(np.flipud(img), cv2.COLOR_RGB2BGR)
        
        cv2.imshow("Door Lock Test", img_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # 让仿真跑一小步，看看物理效果
        env.step(np.zeros(env.action_dim))

    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_load_door()

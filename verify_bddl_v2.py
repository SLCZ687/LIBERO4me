import cv2
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
import os

bddl_file = "libero/libero/bddl_files/my_tasks/KITCHEN_SCENE1_put_the_black_book_in_the_wooden_tray.bddl"

try:
    print("正在启动仿真引擎...")
    # 增加渲染参数
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file, 
        camera_heights=512, 
        camera_widths=512,
        camera_names="agentview"
    )
    
    # reset() 会直接返回初始观测值
    obs = env.reset()
    
    # 在某些版本中，obs 是字典，直接取 image
    img = obs["agentview_image"]
    
    # 保存图片 (注意：robosuite 图像通常是上下颠倒的，需要翻转)
    cv2.imwrite("verify_result.png", img[::-1, :, ::-1])
    print("\n✅ [成功] 环境已加载并截图！")
    print("👉 请检查目录下的 'verify_result.png'")
    env.close()
except Exception as e:
    print(f"\n❌ [截图失败] 环境加载成功但渲染出错: {e}")
    # 如果还是报错，尝试打印 obs 的键看看
    try:
        print(f"可用的观测键值有: {obs.keys()}")
    except:
        pass

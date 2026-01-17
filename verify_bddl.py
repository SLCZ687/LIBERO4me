import cv2
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
import os

# 路径指向你之前生成的 BDDL
bddl_file = "libero/libero/bddl_files/my_tasks/KITCHEN_SCENE1_put_the_black_book_in_the_wooden_tray.bddl"

if not os.path.exists(bddl_file):
    print(f"❌ 找不到文件: {bddl_file}")
else:
    try:
        print("正在启动仿真引擎并加载场景，请稍候...")
        env = OffScreenRenderEnv(bddl_file_name=bddl_file, camera_heights=512, camera_widths=512)
        env.reset()
        # 获取机器人视角的图像
        obs = env._get_observations()
        img = obs["agentview_image"]
        # 保存图片
        cv2.imwrite("verify_result.png", img[::-1, :, ::-1])
        print("\n✅ [成功] 环境已成功加载！")
        print("👉 请在你的文件夹列表中找到并打开 'verify_result.png' 看看画面。")
        env.close()
    except Exception as e:
        print(f"\n❌ [失败] 无法加载 BDDL，错误如下:\n{e}")

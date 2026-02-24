import os
import glob
import cv2
import numpy as np
from libero.libero.envs import OffScreenRenderEnv
import pathlib

CAMERAS = [
    "agentview", 
    # "robot0_eye_in_hand"
    ]
CAMERA_SIZE = (1280, 960)
absolute_path = pathlib.Path(__file__).parent.parent.absolute()
# 目标文件夹路径
bddl_folder = os.path.join(absolute_path, "libero/libero/bddl_files/custom")
# 保存图片的文件夹路径
save_folder = os.path.join(absolute_path, f"custom_scenes_images/{CAMERA_SIZE[0]}x{CAMERA_SIZE[1]}")
    
def main():    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # 获取所有的 bddl 文件
    bddl_files = glob.glob(os.path.join(bddl_folder, "*.bddl"))
    print(f"Found {len(bddl_files)} BDDL files in {bddl_folder}")

    for bddl_file in bddl_files:
        task_name = os.path.splitext(os.path.basename(bddl_file))[0]
        print(f"Processing {task_name}...")
        
        # 环境参数配置
        env_args = {
            "bddl_file_name": bddl_file,
            "camera_heights": CAMERA_SIZE[1],
            "camera_widths": CAMERA_SIZE[0],
            "camera_names": CAMERAS,
        }

        try:
            # 初始化离屏渲染环境
            env = OffScreenRenderEnv(**env_args)
            obs = env.reset()
            
            # 跳过刚开始模拟的前几帧画面（例如跳过 10 帧）
            for _ in range(10):
                obs, _, _, _ = env.step([0.0] * 7)
            
            # 获取相机画面
            images_to_concat = []
            for cam in CAMERAS:
                img = obs[f"{cam}_image"]
                images_to_concat.append(img)
            
            # 将多个相机的画面垂直拼接在一起
            images = np.concatenate(images_to_concat, axis=0)
            
            # robosuite 返回的图像是上下颠倒的，并且是 RGB 格式
            # OpenCV 需要 BGR 格式，所以我们需要翻转图像并转换通道
            # [::-1, :, ::-1] 表示：
            # 第一个 ::-1 翻转高度（上下颠倒）
            # 第二个 : 保持宽度不变
            # 第三个 ::-1 翻转通道（RGB 转 BGR）
            images = images[::-1, :, ::-1]
            
            # 保存图片
            output_path = os.path.join(save_folder, f"{task_name}.png")
            cv2.imwrite(output_path, images)
            print(f"Saved image to {output_path}")
            
            env.close()
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Failed to process {task_name}: {e}")

if __name__ == "__main__":
    main()

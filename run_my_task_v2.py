import os
import cv2
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.benchmark import get_benchmark_dict

class MyCustomBenchmark(benchmark.Benchmark):
    def __init__(self):
        super().__init__()
        self.task_names = ["KITCHEN_SCENE1_put_the_black_book_in_the_wooden_tray"]

    def get_task_bddl_file(self, task_id):
        return "libero/libero/bddl_files/my_tasks/KITCHEN_SCENE1_put_the_black_book_in_the_wooden_tray.bddl"

# 注册到内存
benchmark_dict = benchmark.get_benchmark_dict()
benchmark_dict["my_custom_benchmark"] = MyCustomBenchmark

print("✅ 内存注册成功！")

try:
    benchmark_instance = benchmark_dict["my_custom_benchmark"]()
    bddl_file = benchmark_instance.get_task_bddl_file(0)
    
    print(f"🚀 正在验证官方流程加载 BDDL...")

    env_args = {
        "bddl_file_name": bddl_file,
        "camera_heights": 512,
        "camera_widths": 512,
        "camera_names": "agentview",
    }

    env = OffScreenRenderEnv(**env_args)
    
    # 关键修改：直接从 reset() 的返回值获取初始观测
    obs = env.reset()
    
    # 兼容性检查：尝试不同的键名获取图像
    img = None
    for key in ["agentview_image", "image"]:
        if key in obs:
            img = obs[key]
            break
            
    if img is not None:
        output_name = "final_standard_verify.png"
        cv2.imwrite(output_name, img[::-1, :, ::-1])
        print(f"✨【终极成功】图片已通过官方 API 流程保存至: {output_name}")
    else:
        print(f"❌ 观测值中找不到图像。可用键值: {list(obs.keys())}")

    env.close()

except Exception as e:
    print(f"❌ 运行中出现错误: {e}")

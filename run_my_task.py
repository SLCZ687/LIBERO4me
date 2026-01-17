import os
import cv2
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.benchmark import get_benchmark_dict

# 1. 手动定义你的自定义 Benchmark 类，绕过文件读取不到的问题
class MyCustomBenchmark(benchmark.Benchmark):
    def __init__(self):
        super().__init__()
        # 这里的名字必须和你的 .bddl 文件名（不含后缀）完全匹配
        self.task_names = ["KITCHEN_SCENE1_put_the_black_book_in_the_wooden_tray"]

    def get_task_bddl_file(self, task_id):
        # 直接返回你的 BDDL 路径
        return "libero/libero/bddl_files/my_tasks/KITCHEN_SCENE1_put_the_black_book_in_the_wooden_tray.bddl"

# 2. 强行注入到官方字典，解决 KeyError
# 获取字典引用
benchmark_dict = benchmark.get_benchmark_dict()
# 注入你的自定义类
benchmark_dict["my_custom_benchmark"] = MyCustomBenchmark

print("✅ 已在内存中成功注册 'my_custom_benchmark'")

# 3. 按照官方标准流程运行
try:
    # 实例化你刚注册的类
    benchmark_instance = benchmark_dict["my_custom_benchmark"]()
    task_id = 0
    bddl_file = benchmark_instance.get_task_bddl_file(task_id)
    
    print(f"🚀 正在加载任务 BDDL: {bddl_file}")

    env_args = {
        "bddl_file_name": bddl_file,
        "camera_heights": 512,
        "camera_widths": 512,
    }

    env = OffScreenRenderEnv(**env_args)
    env.reset()
    
    # 获取图像
    obs = env._get_observations()
    img = obs["agentview_image"]
    
    # 保存结果
    output_name = "final_official_verify.png"
    cv2.imwrite(output_name, img[::-1, :, ::-1])
    
    print(f"✨ 验证完成！图片已保存为: {output_name}")
    env.close()

except Exception as e:
    print(f"❌ 运行中出现错误: {e}")

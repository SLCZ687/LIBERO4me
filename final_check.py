from libero.libero.benchmark import get_benchmark_dict
from libero.libero.envs import OffScreenRenderEnv
import os

# 1. 检查 map 文件里的定义
try:
    from libero.libero.benchmark.libero_suite_task_map import libero_task_map
    tasks = libero_task_map.get("my_custom_benchmark", [])
    print(f"📍 在 map 文件中找到的任务: {tasks}")
except Exception as e:
    print(f"❌ 读取 map 文件失败: {e}")

# 2. 尝试直接加载你的 BDDL 渲染一帧
bddl_path = "libero/libero/bddl_files/my_tasks/KITCHEN_SCENE1_put_the_black_book_in_the_wooden_tray.bddl"
if os.path.exists(bddl_path):
    print("🚀 正在进行最后的物理验证...")
    try:
        env = OffScreenRenderEnv(bddl_file_name=bddl_path)
        env.reset()
        print("✅ [物理层] 成功：BDDL 文件可以被环境正常解析！")
        env.close()
    except Exception as e:
        print(f"❌ [物理层] 失败：{e}")

# 3. 检查初始化文件
pkl_path = "libero/libero/init_files/KITCHEN_SCENE1_put_the_black_book_in_the_wooden_tray.pkl"
if os.path.exists(pkl_path):
    print("✅ [数据层] 成功：初始状态文件 (.pkl) 已就绪！")
else:
    print("❌ [数据层] 失败：找不到 .pkl 文件")

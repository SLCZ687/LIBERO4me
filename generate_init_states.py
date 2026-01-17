import os
import pickle
import numpy as np
from libero.libero.envs import OffScreenRenderEnv

# 使用你验证成功的 BDDL
bddl_file = "libero/libero/bddl_files/my_tasks/KITCHEN_SCENE1_put_the_black_book_in_the_wooden_tray.bddl"
save_dir = "libero/libero/init_files"
save_path = os.path.join(save_dir, "KITCHEN_SCENE1_put_the_black_book_in_the_wooden_tray.pkl")

try:
    print("🚀 开始采样初始状态...")
    env = OffScreenRenderEnv(bddl_file_name=bddl_file)
    
    init_states = []
    num_samples = 20  # 采样20个不同的开局
    
    for i in range(num_samples):
        env.reset()
        # 提取当前场景中所有物体的物理状态坐标
        state = env.sim.get_state().flatten()
        init_states.append(state)
        print(f"已完成采样: {i+1}/{num_samples}")

    os.makedirs(save_dir, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(init_states, f)
    
    print(f"\n✅ [大功告成] 初始状态已保存至: {save_path}")
    env.close()
except Exception as e:
    print(f"\n❌ [采样失败]: {e}")

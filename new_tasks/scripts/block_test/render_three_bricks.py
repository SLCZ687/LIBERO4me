# 关键：先 import task，触发 @register_mu + object 注册
import new_tasks.scripts.block_test.three_brick_scene_task  # noqa: F401

import scripts.test_offscreen_video as tov

if __name__ == "__main__":
    tov.main()

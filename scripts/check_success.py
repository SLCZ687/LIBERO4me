# scripts/check_success.py
import libero.libero.envs.objects.custom_objects
from libero.libero.envs import OffScreenRenderEnv

# 改成你当前要检查的 BDDL
BDDL_PATH = (
    "new_tasks/scripts/block_test/tmp_pddl_files/"
    "THREE_BRICK_SCENE_stack_three_brick_cubes.bddl"
)

env = OffScreenRenderEnv(
    bddl_file_name=BDDL_PATH,
    camera_heights=128,
    camera_widths=128,
)

env.reset()

success = env.check_success()
print("success =", success)

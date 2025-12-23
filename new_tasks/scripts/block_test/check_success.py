# new_tasks/scripts/block_test/check_success.py

import numpy as np


def get_body_z(sim, body_name):
    """
    返回 body 的世界坐标 z
    """
    body_id = sim.model.body_name2id(body_name)
    return sim.data.xpos[body_id][2]


def is_on_top(sim, top_obj, bottom_obj, xy_thresh=0.03, z_thresh=0.02):
    """
    判断 top_obj 是否在 bottom_obj 上
    """
    top_id = sim.model.body_name2id(top_obj)
    bot_id = sim.model.body_name2id(bottom_obj)

    top_pos = sim.data.xpos[top_id]
    bot_pos = sim.data.xpos[bot_id]

    # XY 对齐
    xy_dist = np.linalg.norm(top_pos[:2] - bot_pos[:2])
    if xy_dist > xy_thresh:
        return False

    # Z 高度关系
    z_diff = top_pos[2] - bot_pos[2]
    if z_diff < z_thresh:
        return False

    return True


def check_success(env):
    """
    任务成功条件：
    brick_cube_1 在 brick_cube_2 上
    brick_cube_2 在 brick_cube_3 上
    """
    sim = env.sim

    ok_1 = is_on_top(sim, "brick_cube_1_main", "brick_cube_2_main")
    ok_2 = is_on_top(sim, "brick_cube_2_main", "brick_cube_3_main")

    return ok_1 and ok_2

import numpy as np

def _pos(env, body_name: str) -> np.ndarray:
    bid = env.sim.model.body_name2id(body_name)
    return env.sim.data.body_xpos[bid].copy()

def check_stack3_loose(
    env,
    hold_steps: int = 15,
    xy_tol: float = 0.12,
    min_z_gap: float = 0.02,
):
    """
    宽松版：只要 A 在 B 上、B 在 C 上（允许一定偏移），并且高度差别是“叠起来”的，就算成功。
    body 名用 *_main（与你的 xml/robosuite 加载方式一致）。
    """
    A = _pos(env, "brick_cube_1_main")  # red
    B = _pos(env, "brick_cube_2_main")  # green
    C = _pos(env, "brick_cube_3_main")  # blue

    def above(p, q):
        # p 在 q 上：xy 接近 + z 明显更高
        return (np.linalg.norm(p[:2] - q[:2]) <= xy_tol) and ((p[2] - q[2]) >= min_z_gap)

    ok = above(A, B) and above(B, C)

    # 连续 hold_steps 帧都 ok 才返回 True（防抖）
    if not hasattr(env, "_stack3_hold"):
        env._stack3_hold = 0

    if ok:
        env._stack3_hold += 1
    else:
        env._stack3_hold = 0

    return env._stack3_hold >= hold_steps

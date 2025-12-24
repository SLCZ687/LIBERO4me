import numpy as np


def _body_pos(env, body_name: str):
    bid = env.sim.model.body_name2id(body_name)
    return env.sim.data.body_xpos[bid].copy()


def _body_vel(env, body_name: str):
    # cvel: [angular(3), linear(3)] in global frame for body
    bid = env.sim.model.body_name2id(body_name)
    cvel = env.sim.data.cvel[bid].copy()
    lin = cvel[3:6]
    return lin


def _contacts_between_prefix(env, prefix_a: str, prefix_b: str) -> bool:
    """
    True if there exists any contact between geoms whose names start with prefix_a / prefix_b.
    Example prefix: "brick_cube_1_" "gripper0_"
    """
    m = env.sim.model
    d = env.sim.data
    for i in range(d.ncon):
        c = d.contact[i]
        g1 = c.geom1
        g2 = c.geom2
        n1 = m.geom_id2name(g1)
        n2 = m.geom_id2name(g2)
        if n1 is None or n2 is None:
            continue
        ok1 = n1.startswith(prefix_a) and n2.startswith(prefix_b)
        ok2 = n1.startswith(prefix_b) and n2.startswith(prefix_a)
        if ok1 or ok2:
            return True
    return False


def _xy_close(p_top, p_bottom, xy_tol: float) -> bool:
    return np.linalg.norm((p_top - p_bottom)[:2]) <= xy_tol


def check_stack3_strict(
    env,
    # 几何阈值：允许有一点偏移
    xy_tol=0.030,
    min_z_gap=0.035,      # A 相对 B、B 相对 C 的最小高度差（略大于方块边长 0.05? 你按实际size调）
    max_z_gap=0.080,      # 最大高度差
    # 稳定阈值
    vel_tol=0.03,         # m/s，越小越严格
    hold_steps=10,        # 连续满足 hold_steps 才返回 True
    # 命名：你的body/geom前缀
    A_body="brick_cube_1_main",
    B_body="brick_cube_2_main",
    C_body="brick_cube_3_main",
    # 夹爪 geom 前缀（robosuite 常见是 gripper0_ 开头）
    gripper_prefix="gripper0_",
):
    """
    严格 success:
    - A/B/C 位置满足堆叠关系（允许偏移）
    - A-B 与 B-C 必须有 contact
    - 夹爪与 A 不能有 contact（否则说明还夹着/顶着）
    - A、B 线速度要小（稳定）
    - 连续 hold_steps 帧满足才算成功
    """
    # init persistent counter on env
    if not hasattr(env, "_stack_hold"):
        env._stack_hold = 0

    pA = _body_pos(env, A_body)
    pB = _body_pos(env, B_body)
    pC = _body_pos(env, C_body)

    # 1) 几何关系
    ok_xy_AB = _xy_close(pA, pB, xy_tol)
    ok_xy_BC = _xy_close(pB, pC, xy_tol)
    dz_AB = pA[2] - pB[2]
    dz_BC = pB[2] - pC[2]
    ok_z_AB = (min_z_gap <= dz_AB <= max_z_gap)
    ok_z_BC = (min_z_gap <= dz_BC <= max_z_gap)

    if not (ok_xy_AB and ok_xy_BC and ok_z_AB and ok_z_BC):
        env._stack_hold = 0
        return False

    # 2) 必须接触：A-B, B-C
    # brick cube geoms 在你 xml 里会被重命名成类似 brick_cube_1_brick_cube_col / ..._vis
    # 所以 prefix 用 "brick_cube_1_" 这种更稳
    ok_contact_AB = _contacts_between_prefix(env, "brick_cube_1_", "brick_cube_2_")
    ok_contact_BC = _contacts_between_prefix(env, "brick_cube_2_", "brick_cube_3_")
    if not (ok_contact_AB and ok_contact_BC):
        env._stack_hold = 0
        return False

    # 3) 关键：夹爪不能碰 A（防止“夹着撞一下就成功”）
    touching_gripper_A = _contacts_between_prefix(env, gripper_prefix, "brick_cube_1_")
    if touching_gripper_A:
        env._stack_hold = 0
        return False

    # 4) 稳定：速度小
    vA = np.linalg.norm(_body_vel(env, A_body))
    vB = np.linalg.norm(_body_vel(env, B_body))
    if vA > vel_tol or vB > vel_tol:
        env._stack_hold = 0
        return False

    # 5) 连续帧 hold
    env._stack_hold += 1
    return env._stack_hold >= hold_steps
import numpy as np
import os
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
import robosuite.utils.transform_utils as T

class BridgeBuildingEnv(SingleArmEnv):
    def __init__(
        self,
        robots,
        gap_size=0.15,
        num_blocks=3,
        block_size=(0.15, 0.03, 0.04), 
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1., 5e-3, 1e-4),
        use_camera_obs=True,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
    ):
        self.gap_size = gap_size
        self.num_blocks = num_blocks
        self.block_dims = block_size
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.platform_half_size = [0.10, 0.15, 0.05] 

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
        )

    def _load_model(self):
        super()._load_model()

        self.robots[0].robot_model.set_base_xpos([0, 0, 0])

        # 1. 桌子
        self.mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=(0, 0, 0.8),
        )
        self.mujoco_arena.set_origin([0.4, 0, 0])

        # 2. 材质定义
        tex_attrib = {"type": "cube"}
        
        # 木块材质
        self.brick_mat = CustomMaterial(
            texture="WoodRed", 
            tex_name="brick_tex",
            mat_name="brick_mat",
            tex_attrib=tex_attrib,
            mat_attrib={"shininess": 0.5, "specular": 0.5, "reflectance": 0.1},
            shared=True,
        )
        
        # 平台材质 (棋盘格)
        self.platform_mat = CustomMaterial(
            texture="PlasterGray", 
            tex_name="platform_tex",
            mat_name="platform_mat",
            tex_attrib={
                "type": "2d", 
                "builtin": "checker", 
                "width": 512, "height": 512, 
                "rgb1": [0.1, 0.1, 0.1], 
                "rgb2": [0.2, 0.2, 0.2]
            },
            mat_attrib={"shininess": 0.1, "specular": 0.1},
            shared=True,
        )

        # ==========================================
        # 3. 创建物体 (格式修复版)
        # ==========================================
        # solref: [time_const, damp_ratio]
        # [0.002, 1] 表示极硬接触，且临界阻尼(不弹)
        
        self.platform_left = BoxObject(
            name="platform_left",
            size=self.platform_half_size,
            material=self.platform_mat,
            rgba=[1, 1, 1, 1],
            friction=[1.0, 0.005, 0.0001],
            solref=[0.002, 1.0],  # 【修复】使用列表
            solimp=[0.99, 0.999, 0.001], # 【修复】使用列表
            joints=None 
        )
        self.platform_right = BoxObject(
            name="platform_right",
            size=self.platform_half_size,
            material=self.platform_mat,
            rgba=[1, 1, 1, 1],
            friction=[1.0, 0.005, 0.0001],
            solref=[0.002, 1.0],
            solimp=[0.99, 0.999, 0.001],
            joints=None 
        )

        self.bricks = []
        for i in range(self.num_blocks):
            brick = BoxObject(
                name=f"brick_{i}",
                size=[x/2 for x in self.block_dims],
                rgba=[1, 0.8, 0.8, 1], 
                material=self.brick_mat,
                density=800,
                friction=[1.5, 0.005, 0.0001], # 增加摩擦力
                solref=[0.002, 1.0], # 硬接触
                solimp=[0.99, 0.999, 0.001],
                joints=[dict(type="free", damping="0.001")]
            )
            self.bricks.append(brick)

        objects = [self.platform_left, self.platform_right] + self.bricks
        
        self.model = ManipulationTask(
            mujoco_arena=self.mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=objects,
        )

    def _setup_references(self):
        super()._setup_references()
        self.brick_body_ids = [
            self.sim.model.body_name2id(brick.root_body) for brick in self.bricks
        ]
        self.platform_left_id = self.sim.model.body_name2id("platform_left_main")
        self.platform_right_id = self.sim.model.body_name2id("platform_right_main")

    def _reset_internal(self):
        super()._reset_internal()

        table_top_z = 0.8
        plat_z = table_top_z + self.platform_half_size[2]
        y_offset = (self.gap_size / 2) + self.platform_half_size[1]
        
        self.sim.model.body_pos[self.platform_left_id] = [0.5, -y_offset, plat_z]
        self.sim.model.body_pos[self.platform_right_id] = [0.5, y_offset, plat_z]

        plat_top_z = plat_z + self.platform_half_size[2]
        # 稍微抬高一点，防止初始穿模
        brick_z_center = plat_top_z + (self.block_dims[2] / 2) + 0.02
        
        for i, brick in enumerate(self.bricks):
            bid = self.sim.model.body_name2id(brick.root_body)
            jnt = self.sim.model.body_jntadr[bid]
            qpos_adr = self.sim.model.jnt_qposadr[jnt]
            
            rx = np.random.uniform(-0.01, 0.01)
            ry = np.random.uniform(-0.01, 0.01)
            pos = [0.49 + rx, y_offset + ry - 0.05, brick_z_center + i*0.06]
            quat = T.axisangle2quat([0, 0, np.random.uniform(-1.5, 1.5)])
            
            self.sim.data.qpos[qpos_adr:qpos_adr+3] = pos
            self.sim.data.qpos[qpos_adr+3:qpos_adr+7] = quat

        for _ in range(50):
            self.sim.forward()

    def _get_observations(self, force_update=False):
        return super()._get_observations(force_update)

    def reward(self, action=None):
        return 0

if __name__ == "__main__":
    import imageio
    import os
    
    print("🚀 生成最终场景预览 (Fixed Physics)...")
    
    # 增加 KP 刚度，减少机械臂抖动
    controller_config = {
        "type": "OSC_POSE",
        "input_max": 1, "input_min": -1,
        "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
        "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
        "kp": 250, # 250 是个比较硬的值
        "damping_ratio": 1, 
        "impedance_mode": "fixed",
        "kp_limits": [0, 300], 
        "damping_ratio_limits": [0, 10],
        "uncouple_pos_ori": True, 
        "control_delta": True,
        "interpolation": None, 
        "ramp_ratio": 0.2
    }

    env = BridgeBuildingEnv(
        robots="Panda",
        gap_size=0.05,
        num_blocks=1,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        camera_names="agentview",
        control_freq=20,
        controller_configs=controller_config
    )
    env.reset()
    
    frames = []
    # 往下压测试硬度
    action = [0, 0, -0.1, 0, 0, 0, -1]
    
    for i in range(30):
        obs, _, _, _ = env.step(action)
        img = obs["agentview_image"][::-1, ::-1]
        frames.append(img)
        
    imageio.mimsave("bridge_hard_v2.mp4", frames, fps=10)
    print(f"✅ 预览视频已保存: {os.path.abspath('bridge_hard_v2.mp4')}")
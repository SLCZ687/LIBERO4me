import numpy as np
import robosuite as suite
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BallObject, MujocoXMLObject
from robosuite.models.tasks import ManipulationTask
from robosuite.environments.base import register_env
import os

# 引入生成器
import generate_maze

# 全局参数
BALL_RADIUS = generate_maze.BALL_RADIUS 

# ==========================================
# 1. 定义迷宫模型
# ==========================================
class MazeObject(MujocoXMLObject):
    def __init__(self, name):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        xml_path = os.path.join(current_dir, "maze.xml")
        
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XML file not found at: {xml_path}")

        super().__init__(
            xml_path,
            name=name,
            # [关键修改] joints=None 表示这是一个静态物体，固定在空间中，推不动
            joints=None,
            obj_type="all",
            duplicate_collision_geoms=True,
        )

# ==========================================
# 2. 定义任务环境类
# ==========================================
class MazeNavigationTask(SingleArmEnv):
    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        use_camera_obs=True,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="agentview",
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
        print(">>> 正在根据 maze_def.txt 重新生成迷宫和路径...")
        generate_maze.generate_maze_files(
            def_file="maze_def.txt", 
            xml_file="maze.xml", 
            path_file="maze_path.npy"
        )
        print(">>> 生成完成。")

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

        # 1. 桌子 (大尺寸)
        self.mujoco_arena = TableArena(
            table_full_size=(0.8, 0.8, 0.05),
            table_offset=(0, 0, 0.8),
        )

        # 2. 迷宫
        self.maze = MazeObject(name="maze")

        # 3. 球
        self.ball = BallObject(
            name="ball",
            size=[BALL_RADIUS], 
            rgba=[0, 1, 0, 1], 
        )

        robot = self.robots[0]
        # [关键修改] 机器人 Z=0.0，放在地面上
        robot.robot_model.set_base_xpos([-0.75, 0, 0.0])

        self.model = ManipulationTask(
            mujoco_arena=self.mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.maze, self.ball],
        )

    def _setup_references(self):
        super()._setup_references()
        self.ball_body_id = self.sim.model.body_name2id(self.ball.root_body)
        
        # [关键修改] 获取迷宫的 Body ID，用于后续强制设置位置
        self.maze_body_id = self.sim.model.body_name2id(self.maze.root_body)
        
        try:
            self.waypoints = np.load("maze_path.npy")
        except FileNotFoundError:
            raise RuntimeError("maze_path.npy 未找到！请确保 generate_maze.py 已运行。")

        diffs = self.waypoints[1:] - self.waypoints[:-1]
        self.segments_len = np.linalg.norm(diffs, axis=1)
        self.cumulative_len = np.concatenate(([0], np.cumsum(self.segments_len)))
        self.total_len = self.cumulative_len[-1]
        
        print(f"任务路径已加载: {len(self.waypoints)} 个关键点, 总长度 {self.total_len:.3f}m")

    def _reset_internal(self):
        super()._reset_internal()
        
        # [关键修改] 1. 设置迷宫位置 (固定物体)
        # 注意：这修改的是模型的“定义位置”，需要 sim.forward() 生效
        self.sim.model.body_pos[self.maze_body_id] = np.array([0, 0, 0.82])
        
        # 2. 设置球的初始位置 (自动对齐到路径起点 waypoints[0])
        # waypoints 是相对迷宫的，所以我们要加上迷宫的高度 0.82
        start_pos_world = self.waypoints[0] + np.array([0, 0, 0.01 + 0.82]) 
        
        self.sim.data.set_joint_qpos(self.ball.joints[0], np.concatenate([start_pos_world, [1, 0, 0, 0]]))

    def _get_observations(self, force_update=False):
        obs = super()._get_observations(force_update=force_update)
        obs["ball_pos"] = self.sim.data.body_xpos[self.ball_body_id]
        return obs

    def get_progress(self):
        current_pos = self.sim.data.body_xpos[self.ball_body_id]
        
        if current_pos[2] < 0.6: 
            return 0.0
            
        P = current_pos.copy()
        # 修正：相对于迷宫基准高度
        P[2] -= 0.82 

        min_dist_sq = float('inf')
        best_progress_dist = 0.0
        
        for i in range(len(self.waypoints) - 1):
            A = self.waypoints[i]
            B = self.waypoints[i+1]
            
            A_xy = A[:2]; B_xy = B[:2]; P_xy = P[:2]
            
            AB = B_xy - A_xy
            AP = P_xy - A_xy
            
            len_sq = np.dot(AB, AB)
            if len_sq == 0: continue 
            
            t = np.dot(AP, AB) / len_sq
            t = np.clip(t, 0.0, 1.0)
            
            closest = A_xy + t * AB
            dist_sq = np.sum((P_xy - closest)**2)
            
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                best_progress_dist = self.cumulative_len[i] + t * self.segments_len[i]
        
        return np.clip(best_progress_dist / self.total_len, 0.0, 1.0)

    def reward(self, action=None):
        return self.get_progress()

register_env(MazeNavigationTask)
import os
import libero.libero.envs.objects.custom_objects
from libero.libero.utils.mu_utils import register_mu, InitialSceneTemplates
from libero.libero.utils.task_generation_utils import register_task_info, generate_bddl_from_task_info
from libero.libero.utils.bddl_generation_utils import get_xy_region_kwargs_list_from_regions_info

@register_mu(scene_type="kitchen")
class RingConstructionScene(InitialSceneTemplates):
    def __init__(self):
        fixture_num_info = {
            "kitchen_table": 1,
        } 
        
        object_num_info = {
            "ring_stand": 1, # 杆子
            "torus_ring": 1, # 圆环
        }

        super().__init__(
            workspace_name="kitchen_table",
            fixture_num_info=fixture_num_info,
            object_num_info=object_num_info,
        )

    def define_regions(self):
        # 布局参数
        stand_pos_x = -0.15
        stand_pos_y = 0.10
        
        ring_pos_x = 0.0
        ring_pos_y = -0.10
        
        # 1. 杆子初始区域
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[stand_pos_x, stand_pos_y], 
                region_name="stand_init_region", 
                target_name=self.workspace_name, 
                region_half_len=0.01,
            )
        )

        # 2. 圆环初始区域
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[ring_pos_x, ring_pos_y], 
                region_name="ring_init_region", 
                target_name=self.workspace_name, 
                region_half_len=0.05,
            )
        )

        # 3. [关键] 目标判定区域 (在杆子周围)
        # 这个区域应该以杆子的位置为中心。
        # 注意：这里的 target_name 依然是 kitchen_table，因为我们是在桌面上定义绝对坐标区域
        # 只要 Ring 进入这个 XY 范围，且 Z 高度合适，就算成功。
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[stand_pos_x, stand_pos_y],
                region_name="ring_target_region", 
                target_name=self.workspace_name,
                # 范围稍微比杆子粗细大一点，比圆环半径小一点，确保圆环大致居中
                region_half_len=0.04, 
            )
        )
        
        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(self.regions)

    @property
    def init_states(self):
        states = [
            ("On", "ring_stand_1", "kitchen_table_stand_init_region"),
            ("On", "torus_ring_1", "kitchen_table_ring_init_region"),
        ]
        return states

if __name__ == "__main__":
    scene_name = "ring_construction_scene" 
    language = "Pick up the red ring and insert it onto the stand"
    
    # 定义目标状态
    # 1. 圆环在目标区域内 (XY 平面靠近杆子)
    # 2. 圆环和杆子有接触 (可选，但在物理上套进去肯定会有接触或落在底座上)
    # 3. 圆环的高度不能太高 (确保落到底部) -> 这可以通过 InRegion 的 z_range 控制，
    #    但 Libero 默认 InRegion 主要看 XY。
    #    更严格的检查是: On(torus_ring_1, ring_stand_1)
    
    register_task_info(language,
                    scene_name=scene_name,
                    objects_of_interest=["torus_ring_1", "ring_stand_1"],
                    goal_states=[
                        # 核心目标：圆环进入以杆子为中心的区域
                        ("on", "torus_ring_1", "kitchen_table_ring_target_region"),
                        ("on", "torus_ring_1", "ring_stand_1"), 
                    ],
    )

    BDDL_FOLDER = "./custom_pddl"
    bddl_files, failures = generate_bddl_from_task_info(folder=BDDL_FOLDER)
    
    print(f"\nGenerated Task Files: {bddl_files}")
    
    if failures:
        print("Failures:", failures)
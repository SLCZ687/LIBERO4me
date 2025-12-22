import os
import libero.libero.envs.objects.custom_objects
from libero.libero.utils.mu_utils import register_mu, InitialSceneTemplates
from libero.libero.utils.task_generation_utils import register_task_info, generate_bddl_from_task_info
from libero.libero.utils.bddl_generation_utils import get_xy_region_kwargs_list_from_regions_info

@register_mu(scene_type="kitchen")
class BridgeConstructionScene(InitialSceneTemplates):
    def __init__(self):
        fixture_num_info = {
            "kitchen_table": 1,
        } 
        
        object_num_info = {
            "bridge_platform": 2,
            "bridge_brick": 1,
        }

        super().__init__(
            workspace_name="kitchen_table",
            fixture_num_info=fixture_num_info,
            object_num_info=object_num_info,
        )

    def get_region_dict(self, region_centroid_xy, region_name, target_name=None, region_half_len=0.02, yaw_rotation=(0.0, 0.0)):
        if isinstance(region_half_len, (list, tuple)):
            hx, hy = region_half_len
            if target_name is None:
                target_name = self.workspace_name
            
            return {
                region_name: {
                    "target": target_name,
                    "ranges": [
                        (
                            region_centroid_xy[0] - hx, 
                            region_centroid_xy[1] - hy, 
                            region_centroid_xy[0] + hx, 
                            region_centroid_xy[1] + hy, 
                        )
                    ],
                    "yaw_rotation": [yaw_rotation],
                }
            }
        else:
            return super().get_region_dict(region_centroid_xy, region_name, target_name, region_half_len, yaw_rotation)

    def define_regions(self):
        gap_size = 0.09
        platform_len_x = 0.20
        platform_len_y = 0.30 
        y_offset = (gap_size / 2) + (platform_len_y / 2)
        
        bridge_x = -0.05  
        brick_x  = -0.30  

        # 1. 左平台位置
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[bridge_x, -y_offset], 
                region_name="left_platform_init_region", 
                target_name=self.workspace_name, 
                region_half_len=0.001,
            )
        )

        # 2. 右平台位置
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[bridge_x, y_offset], 
                region_name="right_platform_init_region", 
                target_name=self.workspace_name, 
                region_half_len=0.001,
            )
        )

        # 3. 砖块初始位置
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[brick_x, -y_offset], 
                region_name="brick_init_region", 
                target_name=self.workspace_name, 
                region_half_len=[0.02, 0.05], 
            )
        )

        # 4. [关键] 目标区域 (悬浮的 3D 盒子)
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[bridge_x, 0.0],
                region_name="bridge_gap_target_region", 
                target_name=self.workspace_name, 
                region_half_len=[platform_len_x/2, gap_size/2]
            )
        )
        
        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(self.regions)

    @property
    def init_states(self):
        states = [
            ("On", "bridge_platform_1", "kitchen_table_left_platform_init_region"),
            ("On", "bridge_platform_2", "kitchen_table_right_platform_init_region"),
            ("On", "bridge_brick_1", "kitchen_table_brick_init_region")
        ]
        return states

if __name__ == "__main__":
    scene_name = "bridge_construction_scene" 
    language = "Pick up the brick and place it across the gap"
    
    register_task_info(language,
                    scene_name=scene_name,
                    objects_of_interest=["bridge_brick_1"],
                    goal_states=[
                        ("incontact", "bridge_brick_1", "bridge_platform_1"),
                        ("incontact", "bridge_brick_1", "bridge_platform_2")
                    ],
    )

    BDDL_FOLDER = "./custom_pddl"
    bddl_files, failures = generate_bddl_from_task_info(folder=BDDL_FOLDER)
    
    print(f"\nGenerated Task Files: {bddl_files}")
    
    if failures:
        print("Failures:", failures)
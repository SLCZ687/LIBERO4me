import os
import libero.libero.envs.objects.custom_objects
from libero.libero.utils.mu_utils import register_mu, InitialSceneTemplates
from libero.libero.utils.task_generation_utils import register_task_info, generate_bddl_from_task_info
from libero.libero.utils.bddl_generation_utils import get_xy_region_kwargs_list_from_regions_info

@register_mu(scene_type="kitchen")
class SeesawBalanceScene(InitialSceneTemplates):
    def __init__(self):
        fixture_num_info = {
            "kitchen_table": 1,
        } 
        
        object_num_info = {
            "seesaw": 1,
            # "weight_small": 1,
            # "weight_medium_ref": 1,    # 参考物
            "weight_medium_target": 1, # 目标物
            "weight_large": 1,
        }

        super().__init__(
            workspace_name="kitchen_table",
            fixture_num_info=fixture_num_info,
            object_num_info=object_num_info,
        )

    # [关键修复] 重写此方法以支持长方形区域 (list) 和 3D 区域 (z_range)
    def get_region_dict(self, region_centroid_xy, region_name, target_name=None, region_half_len=0.02, yaw_rotation=(0.0, 0.0), z_range=None):
        # 1. 解析长宽
        if isinstance(region_half_len, (list, tuple)):
            hx, hy = region_half_len
        else:
            hx = hy = region_half_len

        if target_name is None:
            target_name = self.workspace_name
            
        # 2. 构建范围列表
        if z_range is not None:
            # 3D 区域: (x_min, y_min, z_min, x_max, y_max, z_max)
            # 这对于检查物体是否悬浮在平衡高度至关重要
            ranges = [
                (
                    region_centroid_xy[0] - hx, 
                    region_centroid_xy[1] - hy, 
                    z_range[0], # z_min
                    region_centroid_xy[0] + hx, 
                    region_centroid_xy[1] + hy,
                    z_range[1]  # z_max
                )
            ]
        else:
            # 2D 区域 (x_min, y_min, x_max, y_max)
            ranges = [
                (
                    region_centroid_xy[0] - hx, 
                    region_centroid_xy[1] - hy, 
                    region_centroid_xy[0] + hx, 
                    region_centroid_xy[1] + hy, 
                )
            ]

        return {
            region_name: {
                "target": target_name,
                "ranges": ranges,
                "yaw_rotation": [yaw_rotation],
            }
        }

    def define_regions(self):
        # 布局参数
        x_offset = -0.2
        y_offset = 0.0
        seesaw_pos = [0.0 + x_offset, 0.20 + y_offset]
        # 托盘离中心
        seesaw_offset_x = 0 
        seesaw_offset_y = 0.16
        
        # 1. 跷跷板放置区域
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=seesaw_pos, 
                region_name="seesaw_init_region", 
                target_name=self.workspace_name, 
                region_half_len=0.01,
            )
        )

        # 2. 跷跷板左托盘初始区域 (用于放置 Reference Block)
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[seesaw_pos[0] - seesaw_offset_x, seesaw_pos[1] - seesaw_offset_y], 
                region_name="left_tray_init_region", 
                target_name=self.workspace_name, 
                region_half_len=0.03,
            )
        )
        
        # 3. 跷跷板右托盘区域 (用于目标判断)
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[seesaw_pos[0] + seesaw_offset_x, seesaw_pos[1] + seesaw_offset_y], 
                region_name="right_tray_region", 
                target_name=self.workspace_name, 
                region_half_len=0.04,
            )
        )

        # 4. 桌面方块备选区域 (分为两个独立区域)
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.05 + x_offset, -0.20], 
                region_name="medium_weights_init_region", 
                target_name=self.workspace_name, 
                region_half_len=0.03, 
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.05 + x_offset, -0.20], 
                region_name="large_weights_init_region", 
                target_name=self.workspace_name, 
                region_half_len=0.03, 
            )
        )

        # # 5. [核心] 平衡高度区域 (Balance Height Region)
        # # 这是一个 3D 区域。只有当跷跷板平衡时，左侧参考物才会处于这个高度。
        # self.regions.update(
        #     self.get_region_dict(
        #         region_centroid_xy=[seesaw_pos[0] - seesaw_offset_x, seesaw_pos[1] - seesaw_offset_y], 
        #         region_name="balance_height_region", 
        #         target_name=self.workspace_name, 
        #         region_half_len=0.05,
        #         # z_range=[0.11, 0.17] # [z_min, z_max]
        #     )
        # )

        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(self.regions)

    @property
    def init_states(self):
        states = [
            ("On", "seesaw_1", "kitchen_table_seesaw_init_region"),
            
            # 参考物放在左边 (会导致左低右高)
            # ("On", "weight_medium_ref_1", "kitchen_table_left_tray_init_region"),
            # ("On", "weight_medium_ref_1", "kitchen_table_weights_init_region"),
            
            # 其他三个放在桌面前面
            # ("On", "weight_small_1", "kitchen_table_weights_init_region"),
            ("On", "weight_medium_target_1", "kitchen_table_medium_weights_init_region"),
            ("On", "weight_large_1", "kitchen_table_large_weights_init_region"),
        ]
        return states

if __name__ == "__main__":
    scene_name = "seesaw_balance_scene" 
    language = "Pick up the two cyan blocks and place them on the ends of the seesaw to make the seesaw low on the left and high on the right"
    
    register_task_info(language,
                    scene_name=scene_name,
                    objects_of_interest=["weight_medium_target_1", "seesaw_1", "weight_large_1"],
                    goal_states=[
                        ("incontact", "weight_medium_target_1", "seesaw_1"),
                        ("incontact", "weight_large_1", "seesaw_1"),
                        ("inregion", "weight_medium_target_1", "kitchen_table_right_tray_region"),
                        ("inregion", "weight_large_1", "kitchen_table_left_tray_init_region"),
                    ],
    )

    BDDL_FOLDER = "./custom_pddl"
    bddl_files, failures = generate_bddl_from_task_info(folder=BDDL_FOLDER)
    
    print(f"\nGenerated Task Files: {bddl_files}")
    if failures:
        print("Failures:", failures)
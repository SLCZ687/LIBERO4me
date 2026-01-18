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
            "bridge_brick": 3,
        }

        super().__init__(
            workspace_name="kitchen_table",
            fixture_num_info=fixture_num_info,
            object_num_info=object_num_info,
        )

    def get_region_dict(self, region_centroid_xy, region_name, target_name=None, region_half_len=0.02, yaw_rotation=(0.0, 0.0), z_range=None):
        if isinstance(region_half_len, (list, tuple)):
            hx, hy = region_half_len
        else:
            hx = hy = region_half_len

        if target_name is None:
            target_name = self.workspace_name
        
        # Default Z range if not provided (table surface) usually handled by "On" predicate but region ranges can include Z.
        # Original BDDL had Z ranges. Let's keep it simple 2D unless needed.
        # "ranges": ((-0.051 -0.196 -0.049 -0.194)) -> very small range? 
        
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
        brick_x_offset = -0.3
        platform_x_offset = -0.05
        # 1. Left Platform Region
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[platform_x_offset, -0.195], 
                region_name="left_platform_init_region", 
                target_name=self.workspace_name, 
                region_half_len=0.002, # Adjust as needed
            )
        )
        
        # 2. Right Platform Region
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[platform_x_offset, 0.195], 
                region_name="right_platform_init_region", 
                target_name=self.workspace_name, 
                region_half_len=0.002,
            )
        )

        # 3. Brick Regions - Define separate regions for each brick
        # Brick 1 Region
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[brick_x_offset, -0.2], 
                region_name="brick_init_region_1", 
                target_name=self.workspace_name, 
                region_half_len=0.02,
            )
        )
        
        # Brick 2 Region
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[brick_x_offset, -0.0], 
                region_name="brick_init_region_2", 
                target_name=self.workspace_name, 
                region_half_len=0.02,
            )
        )
        
        # Brick 3 Region
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[brick_x_offset, 0.2], 
                region_name="brick_init_region_3", 
                target_name=self.workspace_name, 
                region_half_len=0.02,
            )
        )

        # 4. Target Gap Region
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.05, 0.0], 
                region_name="bridge_gap_target_region", 
                target_name=self.workspace_name, 
                region_half_len=[0.07, 0.09], # Approx from BDDL
            )
        )

        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(self.regions)

    @property
    def init_states(self):
        states = [
            ("On", "bridge_platform_1", "kitchen_table_left_platform_init_region"),
            ("On", "bridge_platform_2", "kitchen_table_right_platform_init_region"),
            
            ("On", "bridge_brick_1", "kitchen_table_brick_init_region_1"),
            ("On", "bridge_brick_2", "kitchen_table_brick_init_region_2"),
            ("On", "bridge_brick_3", "kitchen_table_brick_init_region_3"),
        ]
        return states

if __name__ == "__main__":
    scene_name = "bridge_construction_scene" 
    language = "Place the left brick on the left platform, the right brick on the right platform, and the middle brick across the gap on top of the first two bricks"
    
    register_task_info(language,
                    scene_name=scene_name,
                    objects_of_interest=["bridge_brick_1", "bridge_brick_2", "bridge_brick_3", "bridge_platform_1", "bridge_platform_2"],
                    goal_states=[
                        # Left brick on left platform
                        ("incontact", "bridge_brick_1", "bridge_platform_1"),
                        # Right brick on right platform
                        ("incontact", "bridge_brick_3", "bridge_platform_2"),
                        # Middle brick (Brick 2) spans the two
                        ("incontact", "bridge_brick_2", "bridge_brick_1"),
                        ("incontact", "bridge_brick_2", "bridge_brick_3"),
                        # Optional: ensure it's in the gap region
                        ("inregion", "bridge_brick_2", "kitchen_table_bridge_gap_target_region"),
                    ],
    )

    BDDL_FOLDER = "./custom_pddl"
    bddl_files, failures = generate_bddl_from_task_info(folder=BDDL_FOLDER)
    
    print(f"\nGenerated Task Files: {bddl_files}")
    if failures:
        print("Failures:", failures)

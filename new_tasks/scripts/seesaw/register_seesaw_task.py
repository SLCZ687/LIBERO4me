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
            "weight_light": 1,
            "weight_medium": 1,
            "weight_heavy": 1,
            "target_plate": 1,
        }

        super().__init__(
            workspace_name="kitchen_table",
            fixture_num_info=fixture_num_info,
            object_num_info=object_num_info,
        )

    def get_region_dict(self, region_centroid_xy, region_name, target_name=None, region_half_len=0.02, yaw_rotation=(0.0, 0.0), z_range=None, rgba=None):
        if isinstance(region_half_len, (list, tuple)):
            hx, hy = region_half_len
        else:
            hx = hy = region_half_len

        if target_name is None:
            target_name = self.workspace_name
            
        ranges = [
            (
                region_centroid_xy[0] - hx, 
                region_centroid_xy[1] - hy, 
                region_centroid_xy[0] + hx, 
                region_centroid_xy[1] + hy, 
            )
        ]
        
        if z_range is not None:
             ranges = [
                (
                    region_centroid_xy[0] - hx, 
                    region_centroid_xy[1] - hy, 
                    z_range[0], 
                    region_centroid_xy[0] + hx, 
                    region_centroid_xy[1] + hy,
                    z_range[1]
                )
            ]

        region_info = {
            "target": target_name,
            "ranges": ranges,
            "yaw_rotation": [yaw_rotation],
        }
        
        if rgba is not None:
            region_info["rgba"] = rgba

        return {region_name: region_info}

    def define_regions(self):
        x_offset = 0.0
        
        # 1. Seesaw Region
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.0 + x_offset, 0.25], 
                region_name="seesaw_init_region", 
                target_name=self.workspace_name, 
                region_half_len=0.01,
            )
        )

        # 2. Block Init Regions
        self.regions.update(self.get_region_dict([-0.2, -0.2], "light_init_region", region_half_len=0.01))
        self.regions.update(self.get_region_dict([-0.1,  -0.2], "medium_init_region", region_half_len=0.01))
        self.regions.update(self.get_region_dict([0.0,  -0.2], "heavy_init_region",  region_half_len=0.01))

        # 3. Target Plate Region
        self.regions.update(self.get_region_dict([0.1, -0.2], "target_plate_init_region", region_half_len=0.01))

        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(self.regions)

    @property
    def init_states(self):
        states = [
            ("On", "seesaw_1", "kitchen_table_seesaw_init_region"),
            ("On", "weight_light_1", "kitchen_table_light_init_region"),
            ("On", "weight_medium_1", "kitchen_table_medium_init_region"),
            ("On", "weight_heavy_1", "kitchen_table_heavy_init_region"),
            ("On", "target_plate_1", "kitchen_table_target_plate_init_region"),
        ]
        return states

if __name__ == "__main__":
    scene_name = "seesaw_balance_scene" 
    language = "Compare the weights of the three blocks using the seesaw and stack them on the red target plate with heavier blocks at the bottom and lighter blocks on top"
    
    register_task_info(language,
                    scene_name=scene_name,
                    objects_of_interest=["weight_light_1", "weight_medium_1", "weight_heavy_1", "seesaw_1", "target_plate_1"],
                    goal_states=[
                        ("on", "weight_heavy_1", "target_plate_1"),
                        ("on", "weight_medium_1", "weight_heavy_1"),
                        ("on", "weight_light_1", "weight_medium_1"),
                    ],
    )

    BDDL_FOLDER = "./custom_pddl"
    bddl_files, failures = generate_bddl_from_task_info(folder=BDDL_FOLDER)
    
    print(f"\nGenerated Task Files: {bddl_files}")
    if failures:
        print("Failures:", failures)

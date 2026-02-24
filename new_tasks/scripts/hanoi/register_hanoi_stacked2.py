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
            "ring_stand": 1,      # Stand 1
            "ring_stand_two": 1,  # Stand 2
            "ring_stand_three": 1, # Stand 3
            "torus_ring": 1,
            "torus_ring_green": 1,
            "torus_ring_blue": 1,
            "torus_ring_orange": 1,
        }

        super().__init__(
            workspace_name="kitchen_table",
            fixture_num_info=fixture_num_info,
            object_num_info=object_num_info,
        )

    # Helper from seesaw task for flexible regions
    def get_region_dict(self, region_centroid_xy, region_name, target_name=None, region_half_len=0.02, yaw_rotation=(0.0, 0.0), z_range=None):
        if isinstance(region_half_len, (list, tuple)):
            hx, hy = region_half_len
        else:
            hx = hy = region_half_len

        if target_name is None:
            target_name = self.workspace_name
            
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
        else:
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
        x_offset = -0.15
        
        # 1. Stand 1 Region (Start)
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[x_offset, -0.25], 
                region_name="stand_init_region", 
                target_name=self.workspace_name, 
                region_half_len=0.001,
            )
        )
        
        # 2. Stand 2 Region (Goal)
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[x_offset, 0.0], 
                region_name="stand_init_region_2", 
                target_name=self.workspace_name, 
                region_half_len=0.001,
            )
        )

        # 3. Stand 3 Region (Aux)
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[x_offset, 0.25], 
                region_name="stand_init_region_3", 
                target_name=self.workspace_name, 
                region_half_len=0.001,
            )
        )

        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(self.regions)

    @property
    def init_states(self):
        states = [
            # Place Stands
            ("On", "ring_stand_1", "kitchen_table_stand_init_region"),
            ("On", "ring_stand_two_1", "kitchen_table_stand_init_region_2"),
            ("On", "ring_stand_three_1", "kitchen_table_stand_init_region_3"),

            ("On", "torus_ring_orange_1", "ring_stand_1"),
            ("On", "torus_ring_blue_1", "torus_ring_orange_1"),
            ("On", "torus_ring_green_1", "torus_ring_blue_1"),
            ("On", "torus_ring_1", "torus_ring_green_1"),
        ]
        return states

if __name__ == "__main__":
    scene_name = "ring_construction_scene" 
    language = "Move all the rings to the blue stand according to the rules of the Tower of Hanoi, keeping the initial stacked configuration of color in mind. You can only move one ring at a time, and you cannot place a ring initially lower on top of a initially higher ring."
    
    register_task_info(language,
                    scene_name=scene_name,
                    objects_of_interest=["ring_stand_1", "ring_stand_two_1", "ring_stand_three_1", "torus_ring_orange_1", "torus_ring_blue_1", "torus_ring_green_1", "torus_ring_1"],
                    goal_states=[
                        # Goal: Whole stack on Stand 2 (Blue)
                        ("On", "torus_ring_orange_1", "ring_stand_two_1"),
                        ("On", "torus_ring_blue_1", "torus_ring_orange_1"),
                        ("On", "torus_ring_green_1", "torus_ring_blue_1"),
                        ("On", "torus_ring_1", "torus_ring_green_1"),
                    ],
    )

    BDDL_FOLDER = "./custom_pddl"
    bddl_files, failures = generate_bddl_from_task_info(folder=BDDL_FOLDER)
    
    print(f"\nGenerated Task Files: {bddl_files}")
    if failures:
        print("Failures:", failures)

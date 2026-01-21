import os
import libero.libero.envs.objects.custom_objects
from libero.libero.utils.mu_utils import register_mu, InitialSceneTemplates
from libero.libero.utils.task_generation_utils import register_task_info, generate_bddl_from_task_info
from libero.libero.utils.bddl_generation_utils import get_xy_region_kwargs_list_from_regions_info

@register_mu(scene_type="kitchen")
class ShuduScene(InitialSceneTemplates):
    def __init__(self):
        fixture_num_info = {
            "kitchen_table": 1,
        }
        
        # 3 instances of 1, 2, 3
        object_num_info = {
            "external_frame_huanfang": 1,
            "number_block_1": 3,
            "number_block_2": 3,
            "number_block_3": 3,
        }

        super().__init__(
            workspace_name="kitchen_table",
            fixture_num_info=fixture_num_info,
            object_num_info=object_num_info,
        )

    def define_regions(self):
        self.regions = {}
        y_offset = -0.15
        
        # 1. Frame Center Region (Fixed at table center)
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.1, 0.0 + y_offset],
                region_name="frame_center_region",
                target_name=self.workspace_name,
                region_half_len=0.00001
            )
        )

        # 2. Outside Blocks Regions (block_outside_1 to 9)
        # Rows at y=0.25, 0.35, 0.45 (Original) -> 0.10, 0.20, 0.30 (With offset)
        # Cols at x=-0.24, -0.15, -0.06
        outside_coords = [
            (-0.24, 0.25), (-0.15, 0.25), (-0.06, 0.25), # 1, 2, 3
            (-0.24, 0.35), (-0.15, 0.35), (-0.06, 0.35), # 4, 5, 6
            (-0.24, 0.45), (-0.15, 0.45), (-0.06, 0.45)  # 7, 8, 9
        ]
        
        for i, (x, y) in enumerate(outside_coords):
             self.regions.update(
                self.get_region_dict(
                    region_centroid_xy=[x, y + y_offset],
                    region_name=f"block_outside_{i+1}",
                    target_name=self.workspace_name,
                    region_half_len=0.00001
                )
            )

        # 3. Frame Regions (frame_region_1 to 9)
        # Grid 3x3 centered at (-0.1, 0.0) -> (-0.1, -0.15)
        # 1: (-0.2, -0.1), 2: (-0.2, 0.0), 3: (-0.2, 0.1)
        # 4: (-0.1, -0.1), 5: (-0.1, 0.0), 6: (-0.1, 0.1)
        # 7: (0.0, -0.1),  8: (0.0, 0.0),  9: (0.0, 0.1)
        frame_coords = [
            (-0.2, -0.1), (-0.2, 0.0), (-0.2, 0.1),
            (-0.1, -0.1), (-0.1, 0.0), (-0.1, 0.1),
            (0.0, -0.1),  (0.0, 0.0),  (0.0, 0.1)
        ]

        for i, (x, y) in enumerate(frame_coords):
            # Normal region
            self.regions.update(
                self.get_region_dict(
                    region_centroid_xy=[x, y + y_offset],
                    region_name=f"frame_region_{i+1}",
                    target_name=self.workspace_name,
                    region_half_len=0.001 
                )
            )
            # Target region
            self.regions.update(
                self.get_region_dict(
                    region_centroid_xy=[x, y + y_offset],
                    region_name=f"frame_region_{i+1}_target",
                    target_name=self.workspace_name,
                    region_half_len=0.05
                )
            )

        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(self.regions)

    @property
    def init_states(self):
        # Mappings based on Shudu_1.bddl
        # n1_1 -> number_block_1_1 -> frame_region_1
        # n2_1 -> number_block_2_1 -> frame_region_5
        # n1_2 -> number_block_1_2 -> block_outside_1
        # n1_3 -> number_block_1_3 -> block_outside_2
        # n2_2 -> number_block_2_2 -> block_outside_4
        # n2_3 -> number_block_2_3 -> block_outside_5
        # n3_1 -> number_block_3_1 -> block_outside_7
        # n3_2 -> number_block_3_2 -> block_outside_8
        # n3_3 -> number_block_3_3 -> block_outside_9

        states = [
            ("On", "external_frame_huanfang_1", "kitchen_table_frame_center_region"),
            
            # Init Pluaced blocks
            ("On", "number_block_1_1", "kitchen_table_frame_region_1"), 
            ("On", "number_block_2_1", "kitchen_table_frame_region_5"),

            # Outside blocks
            ("On", "number_block_1_2", "kitchen_table_block_outside_1"),
            ("On", "number_block_1_3", "kitchen_table_frame_region_8"),
            ("On", "number_block_2_2", "kitchen_table_frame_region_3"),
            ("On", "number_block_2_3", "kitchen_table_block_outside_5"),
            ("On", "number_block_3_1", "kitchen_table_block_outside_7"),
            ("On", "number_block_3_2", "kitchen_table_frame_region_4"),
            ("On", "number_block_3_3", "kitchen_table_frame_region_9"),
        ]
        return states

if __name__ == "__main__":
    scene_name = "shudu_scene" 
    language = "Place the number blocks in the correct positions to ensure that there are no repeating numbers in any row or column"
    
    register_task_info(language,
                    scene_name=scene_name,
                    objects_of_interest=[
                        "number_block_1_1", "number_block_1_2", "number_block_1_3",
                        "number_block_2_1", "number_block_2_2", "number_block_2_3",
                        "number_block_3_1", "number_block_3_2", "number_block_3_3"
                    ],
                    goal_states=[
                        ("On", "number_block_1_1", "kitchen_table_frame_region_1_target"),
                        ("On", "number_block_1_2", "kitchen_table_frame_region_6_target"),
                        ("On", "number_block_1_3", "kitchen_table_frame_region_8_target"),
                        ("On", "number_block_2_1", "kitchen_table_frame_region_5_target"),
                        ("On", "number_block_2_2", "kitchen_table_frame_region_3_target"),
                        ("On", "number_block_2_3", "kitchen_table_frame_region_7_target"),
                        ("On", "number_block_3_1", "kitchen_table_frame_region_2_target"),
                        ("On", "number_block_3_2", "kitchen_table_frame_region_4_target"),
                        ("On", "number_block_3_3", "kitchen_table_frame_region_9_target"),
                    ],
    )

    BDDL_FOLDER = "./custom_pddl" 
    if os.path.exists("./custom_pddl") is False:
        os.makedirs("./custom_pddl")

    bddl_files, failures = generate_bddl_from_task_info(folder=BDDL_FOLDER)
    
    print(f"\nGenerated Task Files: {bddl_files}")
    if failures:
        print("Failures:", failures)

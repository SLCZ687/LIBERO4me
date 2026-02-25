import os
import libero.libero.envs.objects.custom_objects
from libero.libero.utils.mu_utils import register_mu, InitialSceneTemplates
from libero.libero.utils.task_generation_utils import register_task_info, generate_bddl_from_task_info
from libero.libero.utils.bddl_generation_utils import get_xy_region_kwargs_list_from_regions_info

@register_mu(scene_type="kitchen")
class HuarongdaoScene(InitialSceneTemplates):
    def __init__(self):
        fixture_num_info = {
            "kitchen_table": 1,
        }
        
        object_num_info = {
            "external_frame_huanfang": 1,
            "number_block_1": 1,
            "number_block_2": 1,
            "number_block_3": 1,
            "number_block_4": 1,
            "number_block_5": 1,
            "number_block_6": 1,
            "number_block_7": 1,
            "number_block_8": 1,
        }

        super().__init__(
            workspace_name="kitchen_table",
            fixture_num_info=fixture_num_info,
            object_num_info=object_num_info,
        )

    def define_regions(self):
        self.regions = {}
        x_offset = -0.0
        y_offset = -0.0
        
        # 1. Frame Center Region (Fixed at table center)
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.1 + x_offset, 0.0 + y_offset],
                region_name="frame_center_region",
                target_name=self.workspace_name,
                region_half_len=0.00001 # Minimal range as shown in BDDL
            )
        )

        # 2. Outside Blocks Regions (block_outside_1 to 9)
        # Rows at y=0.25, 0.35, 0.45
        # Cols at x=-0.24, -0.15, -0.06
        outside_coords = [
            (-0.24, 0.25), (-0.15, 0.25), (-0.06, 0.25), # 1, 2, 3
            (-0.24, 0.35), (-0.15, 0.35), (-0.06, 0.35), # 4, 5, 6
            (-0.24, 0.45), (-0.15, 0.45), (-0.06, 0.45)  # 7, 8, 9
        ]
        
        for i, (x, y) in enumerate(outside_coords):
             self.regions.update(
                self.get_region_dict(
                    region_centroid_xy=[x + x_offset, y + y_offset],
                    region_name=f"block_outside_{i+1}",
                    target_name=self.workspace_name,
                    region_half_len=0.0000001 # X range 0.06 wide (-0.03 to +0.03), Y range tiny
                )
            )
             # Note: BDDL ranges for block_outside_1: (-0.27 0.2499999 -0.21 0.2500001)
             # x range: 0.06 -> half_len x = 0.03. y range: tiny.

        # 3. Frame Regions (frame_region_1 to 9)
        # Grid 3x3 centered at (-0.1, 0.0)
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
                    region_centroid_xy=[x + x_offset, y + y_offset],
                    region_name=f"frame_region_{i+1}",
                    target_name=self.workspace_name,
                    region_half_len=0.001 
                )
            )
            # Target region (Larger: 0.1x0.1 -> half_len 0.05)
            # BDDL target example: 1 target (-0.25 -0.15 -0.15 -0.05) -> 0.1 wide. Half len 0.05.
            self.regions.update(
                self.get_region_dict(
                    region_centroid_xy=[x + x_offset, y + y_offset],
                    region_name=f"frame_region_{i+1}_target",
                    target_name=self.workspace_name,
                    region_half_len=0.05
                )
            )

        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(self.regions)

    @property
    def init_states(self):
        states = [
            ("On", "external_frame_huanfang_1", "kitchen_table_frame_center_region"),
            ("On", "number_block_2_1", "kitchen_table_frame_region_1"),
            ("On", "number_block_4_1", "kitchen_table_frame_region_2"),
            ("On", "number_block_1_1", "kitchen_table_frame_region_3"),
            ("On", "number_block_5_1", "kitchen_table_frame_region_4"),
            ("On", "number_block_8_1", "kitchen_table_frame_region_5"),
            ("On", "number_block_3_1", "kitchen_table_frame_region_6"),
            ("On", "number_block_7_1", "kitchen_table_frame_region_7"),
            ("On", "number_block_6_1", "kitchen_table_frame_region_8"),
        ]
        return states

if __name__ == "__main__":
    scene_name = "huarongdao_scene" 
    language = "Slide the blocks to arrange them in ascending order from left to right, top to bottom, leaving the bottom-right corner empty"
    
    register_task_info(language,
                    scene_name=scene_name,
                    objects_of_interest=[
                        "number_block_1_1", "number_block_2_1", "number_block_3_1", 
                        "number_block_4_1", "number_block_5_1", "number_block_6_1", 
                        "number_block_7_1", "number_block_8_1"
                    ],
                    goal_states=[
                        ("On", "number_block_1_1", "kitchen_table_frame_region_1"),
                        ("On", "number_block_2_1", "kitchen_table_frame_region_2"),
                        ("On", "number_block_3_1", "kitchen_table_frame_region_3"),
                        ("On", "number_block_4_1", "kitchen_table_frame_region_4"),
                        ("On", "number_block_5_1", "kitchen_table_frame_region_5"),
                        ("On", "number_block_6_1", "kitchen_table_frame_region_6"),
                        ("On", "number_block_7_1", "kitchen_table_frame_region_7"),
                        ("On", "number_block_8_1", "kitchen_table_frame_region_8"),
                    ],
    )

    # Use a custom folder for output to avoid messing up default ones, or use same as BDDL location
    BDDL_FOLDER = "./custom_pddl" 
    if os.path.exists("./custom_pddl") is False:
        os.makedirs("./custom_pddl")

    bddl_files, failures = generate_bddl_from_task_info(folder=BDDL_FOLDER)
    
    print(f"\nGenerated Task Files: {bddl_files}")
    if failures:
        print("Failures:", failures)

import os
from libero.libero.utils.mu_utils import register_mu, InitialSceneTemplates
from libero.libero.utils.task_generation_utils import register_task_info, generate_bddl_from_task_info
from libero.libero.utils.bddl_generation_utils import get_xy_region_kwargs_list_from_regions_info

@register_mu(scene_type="kitchen")
class MicrowaveScene(InitialSceneTemplates):
    def __init__(self):
        fixture_num_info = {
            "kitchen_table": 1,
            "microwave": 1,
        }

        object_num_info = {
            "white_yellow_mug": 1,
        }

        super().__init__(
            workspace_name="kitchen_table",
            fixture_num_info=fixture_num_info,
            object_num_info=object_num_info,
        )

    def define_regions(self):
        # Microwave region
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.10, 0.35],
                region_name="microwave_init_region",
                target_name=self.workspace_name,
                region_half_len=0.01,
                yaw_rotation=(0, 0),
            )
        )

        # Mug init region
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.15, -0.15],
                region_name="white_yellow_mug_init_region",
                target_name=self.workspace_name,
                region_half_len=0.03,
            )
        )
        
        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(self.regions)

    @property
    def init_states(self):
        states = [
            ("On", "microwave_1", "kitchen_table_microwave_init_region"),
            ("On", "white_yellow_mug_1", "kitchen_table_white_yellow_mug_init_region"),
            ("Close", "microwave_1"), # Start closed
        ]
        return states

if __name__ == "__main__":
    scene_name = "microwave_scene"
    language = "Open the microwave, put the mug inside, and close the microwave"
    
    register_task_info(language,
                       scene_name=scene_name,
                       objects_of_interest=["microwave_1", "white_yellow_mug_1"],
                       goal_states=[
                           ("In", "white_yellow_mug_1", "microwave_1_heating_region"),
                           ("Close", "microwave_1")
                       ]
    )

    # Use a relative directory for output to ensure it's created where expected
    BDDL_FOLDER = "./custom_pddl"
    
    bddl_file_names, failures = generate_bddl_from_task_info(folder=BDDL_FOLDER)
    print("Generated files:", bddl_file_names)
    print("Failures:", failures)

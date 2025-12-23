import numpy as np

# 关键：先 import 注册文件，确保 @register_object 已经执行
import libero.libero.envs.objects.custom_objects  # noqa: F401

from libero.libero.utils.mu_utils import register_mu, InitialSceneTemplates
from libero.libero.utils.bddl_generation_utils import get_xy_region_kwargs_list_from_regions_info
from libero.libero.utils.task_generation_utils import register_task_info, generate_bddl_from_task_info

@register_mu(scene_type="kitchen")
class ThreeBrickScene(InitialSceneTemplates):
    def __init__(self):
        fixture_num_info = {"kitchen_table": 1}
        object_num_info = {"brick_cube": 3}

        super().__init__(
            workspace_name="kitchen_table",
            fixture_num_info=fixture_num_info,
            object_num_info=object_num_info,
        )

    def define_regions(self):
        # 给足空间，避免 RandomizationError
        xs = [-0.18, 0.00, 0.18]
        y = 0.0

        for i, x in enumerate(xs, start=1):
            self.regions.update(
                self.get_region_dict(
                    region_centroid_xy=[x, y],
                    region_name=f"brick_cube_{i}_init_region",
                    target_name=self.workspace_name,
                    region_half_len=0.06,   # 够大，防止采样失败
                )
            )

        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(self.regions)

    @property
    def init_states(self):
        return [
            ("On", "brick_cube_1", "kitchen_table_brick_cube_1_init_region"),
            ("On", "brick_cube_2", "kitchen_table_brick_cube_2_init_region"),
            ("On", "brick_cube_3", "kitchen_table_brick_cube_3_init_region"),
        ]

if __name__ == "__main__":
    scene_name = "three_brick_scene"
    register_task_info(
        "stack three brick cubes",
        scene_name=scene_name,
        objects_of_interest=["brick_cube_1", "brick_cube_2", "brick_cube_3"],
        goal_states=[
            ("On", "brick_cube_1", "brick_cube_2"),
            ("On", "brick_cube_2", "brick_cube_3"),
        ],
    )

    out_dir = "new_tasks/scripts/block_test/tmp_pddl_files"
    bddl_files, failures = generate_bddl_from_task_info(folder=out_dir)
    print("BDDL:", bddl_files)
    print("Failures:", failures)

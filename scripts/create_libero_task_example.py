"""This is a standalone file for create a task in libero."""
import numpy as np

from libero.libero.utils.bddl_generation_utils import (
    get_xy_region_kwargs_list_from_regions_info,
)
from libero.libero.utils.mu_utils import register_mu, InitialSceneTemplates
from libero.libero.utils.task_generation_utils import (
    register_task_info,
    generate_bddl_from_task_info,
)

@register_mu(scene_type="kitchen")
class KitchenScene1(InitialSceneTemplates):
    def __init__(self):
        # 场景中的固定家具
        fixture_num_info = {
            "kitchen_table": 1,
            "wooden_cabinet": 1,
        }

        # 场景中的物体：改为书和托盘
        object_num_info = {
            "black_book": 1,
            "wooden_tray": 1,
        }

        super().__init__(
            workspace_name="kitchen_table",
            fixture_num_info=fixture_num_info,
            object_num_info=object_num_info,
        )

    def define_regions(self):
        # 定义书的初始位置（桌子左侧）
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.15, -0.1],
                region_name="black_book_init_region",
                target_name=self.workspace_name,
                region_half_len=0.03,
            )
        )

        # 定义托盘的初始位置（桌子右侧）
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.15, 0.1],
                region_name="wooden_tray_init_region",
                target_name=self.workspace_name,
                region_half_len=0.03,
            )
        )
        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(
            self.regions
        )

    @property
    def init_states(self):
        # 初始状态：书和托盘都在各自的区域
        states = [
            ("On", "black_book_1", "kitchen_table_black_book_init_region"),
            ("On", "wooden_tray_1", "kitchen_table_wooden_tray_init_region"),
        ]
        return states


def main():
    scene_name = "kitchen_scene1"
    # 这里定义机器人要执行的任务语言
    language = "put the black book in the wooden tray"
    
    register_task_info(
        language,
        scene_name=scene_name,
        objects_of_interest=["black_book_1", "wooden_tray_1"],
        goal_states=[
            # 目标状态：书被放进托盘的堆叠区域
            ("On", "black_book_1", "wooden_tray_1_tray_stack_region"),
        ],
    )
    
    # 核心步骤：运行后会自动生成对应的 .bddl 文件
    bddl_file_names, failures = generate_bddl_from_task_info()
    print(f"成功创建任务 BDDL: {bddl_file_names}")
    if failures:
        print(f"创建失败: {failures}")

if __name__ == "__main__":
    main()
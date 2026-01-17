import numpy as np
from libero.libero.utils.bddl_generation_utils import (
    get_xy_region_kwargs_list_from_regions_info,
)
from libero.libero.utils.mu_utils import register_mu, InitialSceneTemplates
from libero.libero.utils.task_generation_utils import (
    register_task_info,
    generate_bddl_from_task_info,
)

# 1. 定义场景模板：使用更稳定的 wooden_cabinet
@register_mu(scene_type="kitchen")
class KitchenSceneWoodenCabinet(InitialSceneTemplates):
    def __init__(self):
        # 场景中需要的固定家具
        fixture_num_info = {
            "kitchen_table": 1,
        }

        # 场景中需要操作的物体：换成 wooden_cabinet
        object_num_info = {
            "wooden_cabinet": 1, 
        }

        super().__init__(
            workspace_name="kitchen_table",
            fixture_num_info=fixture_num_info,
            object_num_info=object_num_info,
        )

    def define_regions(self):
        # 定义木柜初始化的精确区域
        self.regions.update(
            self.get_region_dict(
                # [0.0, -0.25] 会让柜子放在桌子中心靠后的位置，给机械臂留出开门空间
                region_centroid_xy=[0.0, -0.25],    
                region_name="wooden_cabinet_init_region",
                target_name=self.workspace_name,
                region_half_len=0.005,            # 范围极小，固定位置
                yaw_rotation=(0, 0),              # 角度锁死
            )
        )
        
        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(
            self.regions
        )

    @property
    def init_states(self):
        # 初始化逻辑
        states = [
            ("On", "wooden_cabinet_1", "kitchen_table_wooden_cabinet_init_region"),
        ]
        return states


def main():
    # 关键修改：这里的 scene_name 必须是类名 KitchenSceneWoodenCabinet 的小写形式
    # 即 "kitchenscenewoodencabinet"
    scene_name = "kitchenscenewoodencabinet" 
    language = "open the wooden cabinet"
    
    register_task_info(
        language,
        scene_name=scene_name,
        objects_of_interest=["wooden_cabinet_1"],
        goal_states=[
            ("true",),
        ],
    )

    # 执行生成
    bddl_file_names, failures = generate_bddl_from_task_info()
    
    if bddl_file_names:
        print("\n" + "="*50)
        print(f"成功！生成的 BDDL 路径:\n{bddl_file_names[0]}")
        print("="*50)


    else:
        print("生成失败，请检查是否有语法错误。")

if __name__ == "__main__":
    main()
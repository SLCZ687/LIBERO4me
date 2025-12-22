import numpy as np
import pathlib, re

from libero.libero.utils.mu_utils import register_mu, InitialSceneTemplates
from libero.libero.utils.bddl_generation_utils import get_xy_region_kwargs_list_from_regions_info
from libero.libero.utils.task_generation_utils import register_task_info, generate_bddl_from_task_info

from robosuite.models.objects import MujocoXMLObject
from libero.libero.envs.base_object import register_object
from libero.libero import get_libero_path


class CustomXMLObject(MujocoXMLObject):
    def __init__(self, xml_path, name, joints=None):
        print("[DEBUG] BrickCube XML:", xml_path)
        super().__init__(
            xml_path,
            name=name,
            joints=joints or [],
            obj_type="all",
            duplicate_collision_geoms=False,
        )

        # ======== LIBERO 期望的标准接口（一次性补齐） ========

        # 1. category_name（用于 BDDL / object registry）
        self.category_name = "_".join(
            re.sub(r"([A-Z])", r" \1", self.__class__.__name__).split()
        ).lower()

        # 2. object_properties（至少要有这个 dict）
        self.object_properties = {"vis_site_names": {}}

        # 3. rotation：默认不旋转（但必须存在）
        self.rotation = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
        }

        # 4. rotation_axis：LIBERO 会读，None 是合法的
        self.rotation_axis = None


@register_object
class BrickCube(CustomXMLObject):
    def __init__(self, name="brick_cube", obj_name="brick_cube"):
        # 从当前文件位置，向上找到 repo 根目录
        repo_root = pathlib.Path(__file__).resolve()
        while not (repo_root / "libero").exists():
            repo_root = repo_root.parent

        xml_path = (
            repo_root
            / "libero"
            / "libero"
            / "assets"
            / "custom_objects"
            / "brick_cube"
            / "brick_cube.xml"
        )

        super().__init__(xml_path=str(xml_path), name=name, joints=[])


@register_mu(scene_type="kitchen")
class ThreeBlocksScene(InitialSceneTemplates):
    def __init__(self):
        fixture_num_info = {
            "kitchen_table": 1,
        }

        # 这里用 libero 自带的基础方块对象名（很多版本叫 "cube" / "block" / "wood_block"）
        # 先用 "cube" 作为占位；如果你跑时报 KeyError，我再告诉你用你安装版本里真实的名字。
        object_num_info = {
            "brick_cube": 3,
        }

        super().__init__(
            workspace_name="kitchen_table",
            fixture_num_info=fixture_num_info,
            object_num_info=object_num_info,
        )

    def define_regions(self):
        # 在桌子上 y=0.0 这条线上，x 等间距放三个点
        xs = [-0.18, 0.00, 0.18]   # 拉开间距
        y = 0.0

        for i, x in enumerate(xs, start=1):
            self.regions.update(
                self.get_region_dict(
                    region_centroid_xy=[x, y],
                    region_name=f"brick_cube_{i}_init_region",
                    target_name=self.workspace_name,
                    region_half_len=0.04,   # 放宽采样区域
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
    scene_name = "three_blocks_scene"
    register_task_info(
        "spawn three brick_cubes equally spaced on the table",
        scene_name=scene_name,
        objects_of_interest=["brick_cube_1", "brick_cube_2", "brick_cube_3"],
        goal_states=[],  # 先空，纯测试能不能 reset + 渲染
    )
    
    OUT_DIR = pathlib.Path(__file__).resolve().parent / "tmp_pddl_files"
    bddl_files, failures = generate_bddl_from_task_info(folder=str(OUT_DIR))

    print("BDDL:", bddl_files)
    print("Failures:", failures)

import os
from robosuite.models.objects import MujocoXMLObject
from libero.libero.envs.base_object import register_object
from libero.libero.envs.objects import OBJECTS_DICT
import re
import numpy as np
import pathlib

absolute_path = pathlib.Path(__file__).parent.parent.parent.absolute()

class CustomXmlObject(MujocoXMLObject):
    def __init__(self, folder_name, name, obj_name, joints=[dict(type="free", damping="0.0005")]):
        xml_path = os.path.join(
                str(absolute_path),
                f"assets/custom_objects/{obj_name}/{obj_name}.xml",
            )
        
        super().__init__(
            xml_path,
            name=name,
            joints=joints,
            obj_type="all",
            duplicate_collision_geoms=False,
        )
        
        self.object_properties = {"vis_site_names": {}}
        self.rotation_axis = 'z'
        self.category_name = "_".join(
            re.sub(r"([A-Z])", r" \1", self.__class__.__name__).split()
        ).lower()
        

# --- 砖块 ---
@register_object
class BridgeBrick(CustomXmlObject):
    def __init__(self, name="bridge_brick", obj_name="bridge_brick"):
        super().__init__(
            folder_name="bridge_brick",
            name=name,
            obj_name=obj_name,
            joints=[dict(type="free", damping="0.05")]
        )
        self.rotation = (0, 0)
        
    @property
    def horizontal_radius(self):
        return 0.075
    
    @property
    def bottom_offset(self):
        # 砖块半高 0.02
        # 设置为 -0.03 (抬高 1cm)
        return np.array([0, 0, -0.03])

    @property
    def top_offset(self):
        return np.array([0, 0, 0.02])


# --- 平台 ---
@register_object
class BridgePlatform(CustomXmlObject):
    def __init__(self, name="bridge_platform", obj_name="bridge_platform"):
        super().__init__(
            folder_name="bridge_platform",
            name=name,
            obj_name=obj_name,
            # [关键修改] 阻尼从 2000 降为 50。
            # 50 足够重，能稳住，但允许重力把它拉平，不会卡在半空。
            joints=[dict(type="free", damping="0.05")] 
        )
        self.rotation = (0, 0)
        
    @property
    def horizontal_radius(self):
        return 0.01  
    
    @property
    def bottom_offset(self):
        # 平台半高 0.05
        # 设置为 -0.06 (抬高 1cm，确保完全不穿模)
        return np.array([0, 0, -0.06])
    
    @property
    def top_offset(self):
        return np.array([0, 0, 0.05])

OBJECTS_DICT["bridge_brick"] = BridgeBrick
OBJECTS_DICT["bridge_platform"] = BridgePlatform

# =========================================================
#  Maze Task Objects (Appended)
# =========================================================

# --- 迷宫球 ---
@register_object
class MazeBall(CustomXmlObject):
    def __init__(self, name="maze_ball", obj_name="maze_ball"):
        super().__init__(
            folder_name="maze_ball",
            name=name,
            obj_name=obj_name,
            # [物理属性] 极小阻尼，保证球能顺滑滚动
            joints=[dict(type="free", damping="0.00")]
        )
        self.rotation = (0, 0)
        
    @property
    def horizontal_radius(self):
        # 球半径 0.02
        return 0.02
    
    @property
    def bottom_offset(self):
        # 稍微抬高 0.1mm 防止生成时穿模
        return np.array([0, 0, -0.021])

    @property
    def top_offset(self):
        return np.array([0, 0, 0.02])


# --- 迷宫本体 ---
@register_object
class MazeStructure(CustomXmlObject):
    def __init__(self, name="maze_structure", obj_name="maze_structure"):
        super().__init__(
            folder_name="maze_structure",
            name=name,
            obj_name=obj_name,
            # [物理属性] 巨大阻尼，让迷宫像固定在桌子上一样不动
            joints=[dict(type="free", damping="5000.0")]
        )
        self.rotation = (0, 0)
        
    @property
    def horizontal_radius(self):
        # [关键] 欺骗采样器：虽然迷宫很大，但告诉系统它只有 1cm
        # 这样可以防止放置采样器报 "RandomizationError"
        return 0.01 
    
    @property
    def bottom_offset(self):
        # 迷宫地板很薄，稍微抬高 0.1mm
        return np.array([0, 0, -0.0021])
    
    @property
    def top_offset(self):
        return np.array([0, 0, 0.05])

# [关键] 手动注册 Snake Case 名字映射
# 这样 BDDL 中的 "maze_ball" 和 "maze_structure" 才能被识别
OBJECTS_DICT["maze_ball"] = MazeBall
OBJECTS_DICT["maze_structure"] = MazeStructure


# --- 立方砖 ---
@register_object
class BrickCube(CustomXmlObject):
    def __init__(self, name="brick_cube", obj_name="brick_cube"):
        super().__init__(
            folder_name="brick_cube",
            name=name,
            obj_name=obj_name,
            joints=[dict(type="free", damping="0.05")]
        )
        self.rotation = (0, 0)

    @property
    def horizontal_radius(self):
        # 立方体边长的一半（例如 4cm）
        return 0.02

    @property
    def bottom_offset(self):
        # 半高 + 微抬，防止穿模
        return np.array([0, 0, -0.021])

    @property
    def top_offset(self):
        return np.array([0, 0, 0.02])
OBJECTS_DICT["brick_cube"] = BrickCube


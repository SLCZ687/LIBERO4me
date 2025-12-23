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
        return np.array([0, 0, -0.02])

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
        return 0.18
    
    @property
    def bottom_offset(self):
        # 平台半高 0.05
        return np.array([0, 0, -0.051])
    
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
        # 稍微抬高 1mm 防止生成时穿模
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

# =========================================================
#  Ring Task Objects
# =========================================================

# --- 套圈圆环 ---
@register_object
class TorusRing(CustomXmlObject):
    def __init__(self, name="torus_ring", obj_name="torus_ring"):
        super().__init__(
            folder_name="torus_ring",
            name=name,
            obj_name=obj_name,
            # [物理属性] 较小的阻尼，允许它被推动和调整，
            # 但不要像球那样完全无摩擦，0.005 左右比较像塑料环
            joints=[dict(type="free", damping="0.005")]
        )
        self.rotation = (0, 0)
        
    @property
    def horizontal_radius(self):
        # 圆环整体半径约 0.05 + 管径 0.008 ~= 0.06
        return 0.06
    
    @property
    def bottom_offset(self):
        # 管子半径是 0.008
        # 为了贴合桌面，中心点需要向下偏移半径的距离
        return np.array([0, 0, -0.008])

    @property
    def top_offset(self):
        return np.array([0, 0, 0.008])

# --- 套圈杆子 ---
@register_object
class RingStand(CustomXmlObject):
    def __init__(self, name="ring_stand", obj_name="ring_stand"):
        super().__init__(
            folder_name="ring_stand",
            name=name,
            obj_name=obj_name,
            # [物理属性] 给予极大阻尼 (5000)，模拟沉重的底座
            # 这样机械臂不小心碰到时，它不会轻易飞出去，但从物理引擎角度它仍是可移动物体
            joints=[dict(type="free", damping="5000.0")]
        )
        self.rotation = (0, 0)
        
    @property
    def horizontal_radius(self):
        # 底座是 16cm x 16cm 的方块
        # 半径取 0.08 左右
        return 0.08
    
    @property
    def bottom_offset(self):
        # 底座高度是 0.02 (半高 0.01)
        # 所以底部偏移是 -0.01
        return np.array([0, 0, -0.01])
    
    @property
    def top_offset(self):
        # 杆子高度是 0.15
        return np.array([0, 0, 0.15])

# [关键] 注册到全局字典
# 确保名称与 XML 文件夹及 BDDL 中的名称一致
OBJECTS_DICT["torus_ring"] = TorusRing
OBJECTS_DICT["ring_stand"] = RingStand
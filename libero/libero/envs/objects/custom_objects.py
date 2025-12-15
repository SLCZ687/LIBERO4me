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

print("✅ Custom objects registered: Safe Drop Height + Moderate Damping.")
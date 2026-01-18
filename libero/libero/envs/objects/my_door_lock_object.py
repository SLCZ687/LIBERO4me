import os
import pathlib
import numpy as np
from libero.libero.envs.objects.articulated_objects import ArticulatedObject
from libero.libero.envs.base_object import register_object

# 1. 精确获取你的 assets 文件夹所在的绝对路径
absolute_path = pathlib.Path(__file__).parent.parent.parent.absolute()

@register_object
class MyDoorLock(ArticulatedObject):
    def __init__(
        self,
        name="my_door_lock",
        obj_name="my_door_lock",
        joints=[dict(type="free", damping="0.0005")],
    ):
        # 2. 【核心修改】：手动拼接出你自己的专属路径
        # 指向：libero/libero/assets/my_lab_objects/my_door_lock.xml
        my_xml_path = os.path.join(
            str(absolute_path), f"assets/my_lab_objects/{obj_name}.xml"
        )

        # 3. 调用父类初始化，但传入我们自定义的路径
        # 注意：这里我们直接覆盖了官方 ArticulatedObject 默认的路径逻辑
        from robosuite.models.objects import MujocoXMLObject
        MujocoXMLObject.__init__(
            self,
            my_xml_path, # 使用我们指定的私人路径
            name=name,
            joints=joints,
            obj_type="all",
            duplicate_collision_geoms=False,
        )

        # 4. 剩余的逻辑保持和官方一致
        self.category_name = "my_door_lock"
        self.object_properties = {
            "articulation": {
                "default_open_ranges": [0.5, 1.57],
                "default_close_ranges": [-0.005, 0.0],
            },
            "vis_site_names": {},
        }

        # 必须添加以下属性，供 LIBERO 的采样器使用
        self.rotation = [0, 0] # 允许的旋转范围 [min, max]
        self.rotation_axis = "z" # 绕 Z 轴旋转（即在桌面上平转）
        
        # 顺便检查一下是否有这两个属性，如果没有也加上，防止下一步报错
        if not hasattr(self, "horizontal_radius"):
            self.horizontal_radius = 0.05 # 物体的大致物理半径
        if not hasattr(self, "top_offset"):
            self.top_offset = 0.05 # 物体顶部中心距离底部的偏移

    def is_open(self, qpos):
        return qpos > min(self.object_properties["articulation"]["default_open_ranges"])

    def is_close(self, qpos):
        return qpos < max(self.object_properties["articulation"]["default_close_ranges"])
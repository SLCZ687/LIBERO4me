import os
from robosuite.models.objects import MujocoXMLObject
from libero.libero.envs.base_object import register_object
from libero.libero.envs.objects import OBJECTS_DICT
import re
import numpy as np
import pathlib

absolute_path = pathlib.Path(__file__).parent.parent.parent.absolute()

class CustomXmlObject(MujocoXMLObject):
    def __init__(self, name, obj_name, joints=[dict(type="free", damping="0.0005")]):
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
        self.rotation = (0, 0)
        self.category_name = "_".join(
            re.sub(r"([A-Z])", r" \1", self.__class__.__name__).split()
        ).lower()

#----------------------------------------------------------------------------------------------------------------
#华容道
@register_object
class ExternalFrameHuarongdao(CustomXmlObject):
    def __init__(self, name="external_frame_huarongdao"):
        # 移除 joints=None，让它拥有 free 关节以便环境进行位置初始化
        super().__init__(
            obj_name="external_frame_huarongdao",
            name=name
        )

    @property
    def bottom_offset(self):
        return np.array([0, 0, 0])

    @property
    def top_offset(self):
        return np.array([0, 0, 0])

    @property
    def horizontal_radius(self):
        return 0.12

@register_object
class NumberBlock1(CustomXmlObject):
    def __init__(self, name="number_1"):
        # 对应 assets/custom_objects/number_1/number_1.xml
        super().__init__(
            obj_name="number_1", 
            name=name
        )

    @property
    def bottom_offset(self):
        return np.array([0, 0, -0.02])

    @property
    def top_offset(self):
        return np.array([0, 0, 0.02])

    @property
    def horizontal_radius(self):
        return 0.032

@register_object
class NumberBlock2(CustomXmlObject):
    def __init__(self, name="number_2"):
        # 对应 assets/custom_objects/number_1/number_1.xml
        super().__init__(
            obj_name="number_2", 
            name=name
        )

    @property
    def bottom_offset(self):
        return np.array([0, 0, -0.02])

    @property
    def top_offset(self):
        return np.array([0, 0, 0.02])

    @property
    def horizontal_radius(self):
        return 0.032


@register_object
class NumberBlock3(CustomXmlObject):
    def __init__(self, name="number_3"):
        # 对应 assets/custom_objects/number_1/number_1.xml
        super().__init__(
            obj_name="number_3", 
            name=name
        )

    @property
    def bottom_offset(self):
        return np.array([0, 0, -0.02])

    @property
    def top_offset(self):
        return np.array([0, 0, 0.02])

    @property
    def horizontal_radius(self):
        return 0.032

#--------------------------------------------------------------------------------------------------------------------------
#幻方
@register_object
class ExternalFrameHuanfang(CustomXmlObject):
    def __init__(self, name="external_frame_huanfang"):
        # 移除 joints=None，让它拥有 free 关节以便环境进行位置初始化
        super().__init__(
            obj_name="external_frame_huanfang",
            name=name
        )

    @property
    def bottom_offset(self):
        return np.array([0, 0, 0])

    @property
    def top_offset(self):
        return np.array([0, 0, 0])

    @property
    def horizontal_radius(self):
        return 0.15


@register_object
class NumberBlock4(CustomXmlObject):
    def __init__(self, name="number_4"):
        # 对应 assets/custom_objects/number_1/number_1.xml
        super().__init__(
            obj_name="number_4", 
            name=name
        )

    @property
    def bottom_offset(self):
        return np.array([0, 0, -0.02])

    @property
    def top_offset(self):
        return np.array([0, 0, 0.02])

    @property
    def horizontal_radius(self):
        return 0.032

@register_object
class NumberBlock5(CustomXmlObject):
    def __init__(self, name="number_5"):
        # 对应 assets/custom_objects/number_1/number_1.xml
        super().__init__(
            obj_name="number_5", 
            name=name
        )

    @property
    def bottom_offset(self):
        return np.array([0, 0, -0.02])

    @property
    def top_offset(self):
        return np.array([0, 0, 0.02])

    @property
    def horizontal_radius(self):
        return 0.032


@register_object
class NumberBlock6(CustomXmlObject):
    def __init__(self, name="number_6"):
        # 对应 assets/custom_objects/number_1/number_1.xml
        super().__init__(
            obj_name="number_6", 
            name=name
        )

    @property
    def bottom_offset(self):
        return np.array([0, 0, -0.02])

    @property
    def top_offset(self):
        return np.array([0, 0, 0.02])

    @property
    def horizontal_radius(self):
        return 0.032



@register_object
class NumberBlock7(CustomXmlObject):
    def __init__(self, name="number_7"):
        # 对应 assets/custom_objects/number_1/number_1.xml
        super().__init__(
            obj_name="number_7", 
            name=name
        )

    @property
    def bottom_offset(self):
        return np.array([0, 0, -0.02])

    @property
    def top_offset(self):
        return np.array([0, 0, 0.02])

    @property
    def horizontal_radius(self):
        return 0.032



@register_object
class NumberBlock8(CustomXmlObject):
    def __init__(self, name="number_8"):
        # 对应 assets/custom_objects/number_1/number_1.xml
        super().__init__(
            obj_name="number_8", 
            name=name
        )

    @property
    def bottom_offset(self):
        return np.array([0, 0, -0.02])

    @property
    def top_offset(self):
        return np.array([0, 0, 0.02])

    @property
    def horizontal_radius(self):
        return 0.032




@register_object
class NumberBlock9(CustomXmlObject):
    def __init__(self, name="number_9"):
        # 对应 assets/custom_objects/number_1/number_1.xml
        super().__init__(
            obj_name="number_9", 
            name=name
        )

    @property
    def bottom_offset(self):
        return np.array([0, 0, -0.02])

    @property
    def top_offset(self):
        return np.array([0, 0, 0.02])

    @property
    def horizontal_radius(self):
        return 0.032
    
#-------------------------------------------------------------------------------------------------------------------
@register_object
class ExternalFrameTicTacToe(CustomXmlObject):
    def __init__(self, name="external_frame_tic_tac_toe"):
        # 移除 joints=None，让它拥有 free 关节以便环境进行位置初始化
        super().__init__(
            obj_name="external_frame_tic_tac_toe",
            name=name
        )

    @property
    def bottom_offset(self):
        return np.array([0, 0, 0])

    @property
    def top_offset(self):
        return np.array([0, 0, 0])

    @property
    def horizontal_radius(self):
        return 0.15


@register_object
class BlockO(CustomXmlObject):
    def __init__(self, name="block_o"):
        # 对应 assets/custom_objects/number_1/number_1.xml
        super().__init__(
            obj_name="block_o", 
            name=name
        )

    @property
    def bottom_offset(self):
        return np.array([0, 0, -0.02])

    @property
    def top_offset(self):
        return np.array([0, 0, 0.02])

    @property
    def horizontal_radius(self):
        return 0.032

@register_object
class BlockX(CustomXmlObject):
    def __init__(self, name="block_x"):
        # 对应 assets/custom_objects/number_1/number_1.xml
        super().__init__(
            obj_name="block_x", 
            name=name
        )

    @property
    def bottom_offset(self):
        return np.array([0, 0, -0.02])

    @property
    def top_offset(self):
        return np.array([0, 0, 0.02])

    @property
    def horizontal_radius(self):
        return 0.032
    
#-----------------------------------------------------------------------------------------------------------------------
@register_object
class BlockRed(CustomXmlObject):
    def __init__(self, name="block_red"):
        # 对应 assets/custom_objects/number_1/number_1.xml
        super().__init__(
            obj_name="block_red", 
            name=name
        )

    @property
    def bottom_offset(self):
        return np.array([0, 0, -0.02])

    @property
    def top_offset(self):
        return np.array([0, 0, 0.02])

    @property
    def horizontal_radius(self):
        return 0.032



@register_object
class BlockBlue(CustomXmlObject):
    def __init__(self, name="block_blue"):
        # 对应 assets/custom_objects/number_1/number_1.xml
        super().__init__(
            obj_name="block_blue", 
            name=name
        )

    @property
    def bottom_offset(self):
        return np.array([0, 0, -0.02])

    @property
    def top_offset(self):
        return np.array([0, 0, 0.02])

    @property
    def horizontal_radius(self):
        return 0.032


@register_object
class Basket_my(CustomXmlObject):
    def __init__(self, name="basket_my"):
        # 对应 assets/custom_objects/number_1/number_1.xml
        super().__init__(
            obj_name="basket_my", 
            name=name
        )

    @property
    def bottom_offset(self):
        return np.array([0, 0, -0.02])

    @property
    def top_offset(self):
        return np.array([0, 0, 0.02])

    @property
    def horizontal_radius(self):
        return 0.08

#------------------------------------------------------------------------------------------------------------------
#new part for more bricks
@register_object
class BridgeBrickTwo(CustomXmlObject):
    def __init__(self, name="bridge_brick2", obj_name="bridge_brick2"):
        super().__init__(
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


@register_object
class BridgeBrickThree(CustomXmlObject):
    def __init__(self, name="bridge_brick3", obj_name="bridge_brick3"):
        super().__init__(
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



#-------------------------------------------------------------------------------------------------------------------

# --- 砖块 ---
@register_object
class BridgeBrick(CustomXmlObject):
    def __init__(self, name="bridge_brick", obj_name="bridge_brick"):
        super().__init__(
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
        return np.array([0, 0, -0.002])
    
    @property
    def top_offset(self):
        # 杆子高度是 0.15
        return np.array([0, 0, 0.01])  # 处理初始化时的碰撞问题



#------------------------------------------------------------------------------------------------------------
#new part for more rings
# --- 套圈圆环 ---
@register_object
class TorusRingGreen(CustomXmlObject):
    def __init__(self, name="torus_ring_green", obj_name="torus_ring_green"):
        super().__init__(
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

@register_object
class TorusRingBlue(CustomXmlObject):
    def __init__(self, name="torus_ring_blue", obj_name="torus_ring_blue"):
        super().__init__(
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

@register_object
class TorusRingYellow(CustomXmlObject):
    def __init__(self, name="torus_ring_yellow", obj_name="torus_ring_yellow"):
        super().__init__(
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


@register_object
class TorusRingPurple(CustomXmlObject):
    def __init__(self, name="torus_ring_purple", obj_name="torus_ring_purple"):
        super().__init__(
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


@register_object
class TorusRingPink(CustomXmlObject):
    def __init__(self, name="torus_ring_pink", obj_name="torus_ring_pink"):
        super().__init__(
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
    

@register_object
class TorusRingOrange(CustomXmlObject):
    def __init__(self, name="torus_ring_orange", obj_name="torus_ring_orange"):
        super().__init__(
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

@register_object
class TorusRingSize1(CustomXmlObject):
    def __init__(self, name="torus_ring_size1", obj_name="torus_ring_size1"):
        super().__init__(
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

@register_object
class TorusRingSize2(CustomXmlObject):
    def __init__(self, name="torus_ring_size2", obj_name="torus_ring_size2"):
        super().__init__(
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

@register_object
class TorusRingSize3(CustomXmlObject):
    def __init__(self, name="torus_ring_size3", obj_name="torus_ring_size3"):
        super().__init__(
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
    
@register_object
class TorusRingSize4(CustomXmlObject):
    def __init__(self, name="torus_ring_size4", obj_name="torus_ring_size4"):
        super().__init__(
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
#-------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------
#new part for more stands
# --- 套圈杆子 ---
@register_object
class RingStandTwo(CustomXmlObject):
    def __init__(self, name="ring_stand_two", obj_name="ring_stand2"):
        super().__init__(
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
        return np.array([0, 0, -0.005])
    
    @property
    def top_offset(self):
        # 杆子高度是 0.15
        return np.array([0, 0, 0.01])
    

@register_object
class RingStandThree(CustomXmlObject):
    def __init__(self, name="ring_stand_three", obj_name="ring_stand3"):
        super().__init__(
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
        return np.array([0, 0, -0.005])
    
    @property
    def top_offset(self):
        # 杆子高度是 0.15
        return np.array([0, 0, 0.01])

#-------------------------------------------------------------------------------------------------------

# [关键] 注册到全局字典
# 确保名称与 XML 文件夹及 BDDL 中的名称一致
OBJECTS_DICT["torus_ring"] = TorusRing
OBJECTS_DICT["ring_stand"] = RingStand
OBJECTS_DICT["torus_ring_green"]=TorusRingGreen
OBJECTS_DICT["ring_stand_two"]=RingStandTwo
OBJECTS_DICT["ring_stand_three"]=RingStandThree
OBJECTS_DICT["torus_ring_blue"]=TorusRingBlue
OBJECTS_DICT["torus_ring_orange"]=TorusRingOrange
OBJECTS_DICT["torus_ring_purple"]=TorusRingPurple
OBJECTS_DICT["torus_ring_yellow"]=TorusRingYellow
OBJECTS_DICT["torus_ring_pink"]=TorusRingPink
OBJECTS_DICT["torus_ring_size1"]=TorusRingSize1
OBJECTS_DICT["torus_ring_size2"]=TorusRingSize2
OBJECTS_DICT["torus_ring_size3"]=TorusRingSize3
OBJECTS_DICT["torus_ring_size4"]=TorusRingSize4

#--------------------------------------------------------------------------------------------------------------------------
#---Rectangualr Obstacle
@register_object
class RectangularObstacle(CustomXmlObject):
    def __init__(self, name="rectangular_obstacle", obj_name="rectangular_obstacle"):
        super().__init__(
            name=name,
            obj_name=obj_name,
            joints=[dict(type="free", damping="5.0")]
        )
        self.rotation = (0, 0)
        
    @property
    def horizontal_radius(self):
        return 0.05
    
    @property
    def bottom_offset(self):
        return np.array([0, 0, -0.002])
    
    @property
    def top_offset(self):
        return np.array([0, 0, 0.002])

#---------------------------------------------------------------------------------------------------------------------

# =========================================================
#  Seesaw Task Objects (Appended)
# =========================================================

# --- 跷跷板本体 ---
@register_object
class Seesaw(CustomXmlObject):
    def __init__(self, name="seesaw", obj_name="seesaw"):
        super().__init__(
            name=name,
            obj_name=obj_name,
            # [关键] 阻尼大一点，防止整个底座被推走
            joints=[dict(type="free", damping="5000.0")]
        )
        self.rotation = (np.pi/2, np.pi/2)
        self.rotation_axis = "x"

    @property
    def horizontal_radius(self):
        return 0.25 
    @property
    def bottom_offset(self):
        return np.array([0, 0, -0.06]) 
    @property
    def top_offset(self):
        return np.array([0, 0, 0.12]) 

# --- 1. 轻方块 (Yellow) ---
@register_object
class WeightLight(CustomXmlObject):
    def __init__(self, name="weight_light", obj_name="weight_light"):
        super().__init__(
            name=name,
            obj_name=obj_name,
            joints=[dict(type="free", damping="0.001")]
        )
        self.rotation = (0, 0)

    @property
    def horizontal_radius(self):
        return 0.03 # 0.02 * 1.5 (visual size vs collision) - let's keep it consistent with 4cm block
    @property
    def bottom_offset(self):
        return np.array([0, 0, -0.02])
    @property
    def top_offset(self):
        return np.array([0, 0, 0.02])

# --- 2. 中方块 (Cyan) ---
@register_object
class WeightMedium(CustomXmlObject):
    def __init__(self, name="weight_medium", obj_name="weight_medium"):
        super().__init__(
            name=name,
            obj_name=obj_name,
            joints=[dict(type="free", damping="0.001")]
        )
        self.rotation = (0, 0)

    @property
    def horizontal_radius(self):
        return 0.03 # 0.02 * 1.5
    @property
    def bottom_offset(self):
        return np.array([0, 0, -0.02])
    @property
    def top_offset(self):
        return np.array([0, 0, 0.02])

# --- 3. 重方块 (Purple) ---
@register_object
class WeightHeavy(CustomXmlObject):
    def __init__(self, name="weight_heavy", obj_name="weight_heavy"):
        super().__init__(
            name=name,
            obj_name=obj_name,
            joints=[dict(type="free", damping="0.001")]
        )
        self.rotation = (0, 0)
        
    @property
    def horizontal_radius(self):
        return 0.03 # 0.02 * 1.5
    @property
    def bottom_offset(self):
        return np.array([0, 0, -0.02]) 
    @property
    def top_offset(self):
        return np.array([0, 0, 0.02])

# 手动注册到字典 
OBJECTS_DICT["seesaw"] = Seesaw
OBJECTS_DICT["weight_light"] = WeightLight
OBJECTS_DICT["weight_medium"] = WeightMedium
OBJECTS_DICT["weight_heavy"] = WeightHeavy

# --- 目标底座 (Red Plate) ---
@register_object
class TargetPlate(CustomXmlObject):
    def __init__(self, name="target_plate", obj_name="target_plate"):
        super().__init__(
            name=name,
            obj_name=obj_name,
            joints=[dict(type="free", damping="5000.0")]
        )
        self.rotation = (0, 0)
    
    @property
    def horizontal_radius(self):
        return 0.05
        
    @property
    def bottom_offset(self):
        return np.array([0, 0, -0.0025])
        
    @property
    def top_offset(self):
        return np.array([0, 0, 0.0025])

OBJECTS_DICT["target_plate"] = TargetPlate

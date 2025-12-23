import os
import math
import numpy as np

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

ASSET_ROOT = "custom_assets"
create_folder(ASSET_ROOT)

# =========================================================
# 工具函数：生成圆环的 XML body 内容
# =========================================================
def generate_torus_body(major_radius=0.06, tube_radius=0.008, num_segments=16, color="1 0 0 1"):
    """
    使用 fromto 属性连接顶点，无需计算欧拉角，也不会有单位问题。
    """
    geoms = []
    
    for i in range(num_segments):
        # 当前顶点的角度
        theta1 = 2 * math.pi * i / num_segments
        # 下一个顶点的角度
        theta2 = 2 * math.pi * ((i + 1) % num_segments) / num_segments
        
        # 计算起点 (Start Point)
        x1 = major_radius * math.cos(theta1)
        y1 = major_radius * math.sin(theta1)
        
        # 计算终点 (End Point)
        x2 = major_radius * math.cos(theta2)
        y2 = major_radius * math.sin(theta2)
        
        # 使用 fromto 定义胶囊
        # 注意：使用 fromto 时，size 只需要一个参数 (半径)，长度由起点终点决定
        
        # Collision geom (Group 0)
        geoms.append(
            f'<geom name="ring_seg_{i}" type="capsule" fromto="{x1} {y1} 0 {x2} {y2} 0" size="{tube_radius}" '
            f'rgba="{color}" density="500" friction="1.5 0.005 0.0001" '
            f'solref="0.001 1" solimp="0.95 0.99 0.001" group="0"/>'
        )
        
        # Visual geom (Group 1)
        geoms.append(
            f'<geom name="ring_seg_vis_{i}" type="capsule" fromto="{x1} {y1} 0 {x2} {y2} 0" size="{tube_radius}" '
            f'rgba="{color}" contype="0" conaffinity="0" group="1"/>'
        )

    return "\n        ".join(geoms)

# =========================================================
# 1. Ring Asset (套圈的圈)
# =========================================================
# 定义尺寸
RING_RADIUS = 0.04  # 圆环整体半径
TUBE_RADIUS = 0.008 # 环的粗细
SEGMENTS = 24

ring_body_str = generate_torus_body(RING_RADIUS, TUBE_RADIUS, SEGMENTS, "0.8 0.2 0.2 1")

# 抓取点 Site 的位置：放在圆环的“实体”上，而不是圆心空气处
# 选在 X 轴正方向的那个点上
grasp_site_x = RING_RADIUS
grasp_site_z = 0

ring_xml_content = f"""
<mujoco model="torus_ring">
  <worldbody>
    <body>
      <body name="object">
        <!-- 组合几何体生成圆环 -->
        {ring_body_str}
        
        <!-- Sites -->
        <!-- bottom_site 用于抓取逻辑，放在环的一侧实体上 -->
        <site name="bottom_site" pos="{grasp_site_x} 0 {-TUBE_RADIUS}" size="0.002" rgba="0 0 0 0"/>
        <site name="top_site" pos="{grasp_site_x} 0 {TUBE_RADIUS}" size="0.002" rgba="0 0 0 0"/>
        <!-- center_site 用于判断圆环是否套进了杆子 (位于圆心) -->
        <site name="center_site" pos="0 0 0" size="0.002" rgba="1 0 0 0.5"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

# =========================================================
# 2. Stand Asset (杆子)
# =========================================================
# 底座 + 竖杆
BASE_SIZE = "0.08 0.08 0.005" # 宽底座
POLE_HEIGHT = 0.15
POLE_RADIUS = 0.012

stand_xml_content = f"""
<mujoco model="ring_stand">
  <asset>
    <texture name="tex_stand" type="2d" builtin="flat" rgb1="0.3 0.3 0.3" width="512" height="512"/>
    <material name="mat_stand" texture="tex_stand" shininess="0.5" specular="0.5"/>
  </asset>
  <worldbody>
    <body>
      <body name="object">
        <!-- 底座 (不可移动的基座，稍微重一点) -->
        <geom name="base_geom" type="box" size="{BASE_SIZE}" material="mat_stand" 
              density="50000" friction="2.0 0.005 0.0001" group="0"/>
        <geom name="base_vis" type="box" size="{BASE_SIZE}" material="mat_stand" 
              contype="0" conaffinity="0" group="1"/>
              
        <!-- 竖杆 -->
        <geom name="pole_geom" type="cylinder" size="{POLE_RADIUS} {POLE_HEIGHT/2}" pos="0 0 {POLE_HEIGHT/2 + 0.01}" material="mat_stand"
              density="1000" friction="0.5 0.005 0.0001" group="0"/>
        <geom name="pole_vis" type="cylinder" size="{POLE_RADIUS} {POLE_HEIGHT/2}" pos="0 0 {POLE_HEIGHT/2 + 0.01}" material="mat_stand"
              contype="0" conaffinity="0" group="1"/>

        <!-- 关键 Site: 用于判断圆环是否套到底部 -->
        <!-- 位于杆子根部 -->
        <site name="target_site" pos="0 0 0.02" size="0.002" rgba="0 1 0 0.5"/>
        
        <site name="bottom_site" pos="0 0 -0.01" size="0.002" rgba="0 0 0 0"/>
        <site name="top_site" pos="0 0 {POLE_HEIGHT}" size="0.002" rgba="0 0 0 0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

# 保存文件
ring_dir = os.path.join(ASSET_ROOT, "torus_ring")
create_folder(ring_dir)
with open(os.path.join(ring_dir, "torus_ring.xml"), "w") as f:
    f.write(ring_xml_content)

stand_dir = os.path.join(ASSET_ROOT, "ring_stand")
create_folder(stand_dir)
with open(os.path.join(stand_dir, "ring_stand.xml"), "w") as f:
    f.write(stand_xml_content)

print(f"\n✅ Assets Generated: \n1. {os.path.join(ring_dir, 'torus_ring.xml')} (Composite Capsule Ring)\n2. {os.path.join(stand_dir, 'ring_stand.xml')} (Base + Pole)")
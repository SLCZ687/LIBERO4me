import os
import math

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

ASSET_ROOT = "custom_assets"
create_folder(ASSET_ROOT)

# =========================================================
# 工具函数：生成圆环的 XML body 内容
# =========================================================
def generate_torus_body(major_radius=0.06, tube_radius=0.008, num_segments=16, color="1 0 0 1", collision_radius=None):
    """
    使用 fromto 属性连接顶点，无需计算欧拉角，也不会有单位问题。
    """
    if collision_radius is None:
        collision_radius = tube_radius

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
        # 增加 condim="6" 以启用滚动摩擦支持
        # 增大 friction 的第三个分量 (rolling friction) 防止绕轴滚动下垂
        # 减小 density 降低重量
        # 使用 collision_radius 避免穿模
        geoms.append(
            f'<geom name="ring_seg_{i}" type="capsule" fromto="{x1} {y1} 0 {x2} {y2} 0" size="{collision_radius}" '
            f'rgba="{color}" density="50" friction="1.0 0.05 0.05" condim="6" '
            f'solref="0.001 1" solimp="0.95 0.99 0.001" group="0"/>'
        )
        
        # Visual geom (Group 1)
        geoms.append(
            f'<geom name="ring_seg_vis_{i}" type="capsule" fromto="{x1} {y1} 0 {x2} {y2} 0" size="{tube_radius}" '
            f'rgba="{color}" contype="0" conaffinity="0" group="1"/>'
        )

    return "\n        ".join(geoms)

# =========================================================
# 生成并保存 Torus Ring XML
# =========================================================
def create_ring_xml(name, radius, color):
    # 定义尺寸
    TUBE_RADIUS = 0.012 # 环的粗细
    # 缩小碰撞体半径，防止初始堆叠时因为穿模而弹飞
    COLLISION_RADIUS = TUBE_RADIUS - 0.001 
    SEGMENTS = 24 # 统一用24让它更圆滑

    ring_body_str = generate_torus_body(radius, TUBE_RADIUS, SEGMENTS, color, collision_radius=COLLISION_RADIUS)

    # 抓取点 Site 的位置：放在圆环的“实体”上，而不是圆心空气处
    # 选在 X 轴正方向的那个点上
    grasp_site_x = radius
    
    xml_content = f"""
<mujoco model="{name}">
  <worldbody>
    <body>
      <body name="object">
        <!-- 组合几何体生成圆环 -->
        {ring_body_str}
      </body>
    </body>
  </worldbody>
</mujoco>
"""
    
    # 保存文件
    ring_dir = os.path.join(ASSET_ROOT, name)
    create_folder(ring_dir)
    file_path = os.path.join(ring_dir, f"{name}.xml")
    with open(file_path, "w") as f:
        f.write(xml_content.strip())
    print(f"Generated: {file_path}")

# =========================================================
# 生成并保存 Stand XML
# =========================================================
def create_stand_xml(name, rgb, pole_height):
    # 底座 + 竖杆
    BASE_HALF_HEIGHT = 0.005
    BASE_SIZE_VIS = f"0.08 0.08 {BASE_HALF_HEIGHT}"
    # 碰撞几何体略小，避免悬浮
    BASE_SIZE_COL = f"0.08 0.08 {BASE_HALF_HEIGHT - 0.001}"
    POLE_RADIUS = 0.008
    
    # 计算杆子相关参数
    # Mujoco cylinder size 是半高 (h/2)
    # Pos Z 是中心点。
    # Pole 底部在 Base 顶上
    pole_half_height = pole_height / 2.0
    pole_pos_z = BASE_HALF_HEIGHT + pole_half_height

    xml_content = f"""
<mujoco model="ring_stand">
  <asset>
    <texture name="tex_stand" type="2d" builtin="flat" rgb1="{rgb}" width="512" height="512"/>
    <material name="mat_stand" texture="tex_stand" shininess="0.5" specular="0.5"/>
  </asset>
  <worldbody>
    <body>
      <body name="object">
        <!-- 底座 (不可移动的基座，稍微重一点) -->
        <geom name="base_geom" type="box" size="{BASE_SIZE_COL}" material="mat_stand" 
              density="50000" friction="0.5 0.005 0.0001" group="0"/>
        <geom name="base_vis" type="box" size="{BASE_SIZE_VIS}" material="mat_stand" 
              contype="0" conaffinity="0" group="1"/>
              
        <!-- 竖杆 -->
        <!-- <geom name="pole_geom" type="cylinder" size="{POLE_RADIUS} {pole_half_height}" pos="0 0 {pole_pos_z}" material="mat_stand"
              density="1000" friction="0.5 0.005 0.0001" group="0"/>
        <geom name="pole_vis" type="cylinder" size="{POLE_RADIUS} {pole_half_height}" pos="0 0 {pole_pos_z}" material="mat_stand"
              contype="0" conaffinity="0" group="1"/> -->

        <!-- 关键 Site: 用于判断圆环是否套到底部 -->
        <!-- 位于杆子根部 -->
        <site name="target_site" pos="0 0 0.02" size="0.002" rgba="0 1 0 0.5"/>
        
        <site name="bottom_site" pos="0 0 -0.01" size="0.002" rgba="0 0 0 0"/>
        <site name="top_site" pos="0 0 {pole_height} " size="0.002" rgba="0 0 0 0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""
    
    stand_dir = os.path.join(ASSET_ROOT, name)
    create_folder(stand_dir)
    file_path = os.path.join(stand_dir, f"{name}.xml")
    with open(file_path, "w") as f:
        f.write(xml_content.strip())
    print(f"Generated: {file_path}")


# =========================================================
# Main Generation List
# =========================================================
RING_STAND_HEIGHT = 0.01

# 1. Ring Stands
stands_config = [
    {"name": "ring_stand",  "rgb": "0.3 0.3 0.3", "height": RING_STAND_HEIGHT},
    {"name": "ring_stand2", "rgb": "0.1 0.1 0.8", "height": RING_STAND_HEIGHT},
    {"name": "ring_stand3", "rgb": "0.8 0.1 0.1", "height": RING_STAND_HEIGHT},
]

for s in stands_config:
    create_stand_xml(s["name"], s["rgb"], s["height"])

# 2. Torus Rings
rings_config = [
    {"name": "torus_ring",        "radius": 0.06, "color": "0.8 0.2 0.2 1"}, # Red
    {"name": "torus_ring_blue",   "radius": 0.04, "color": "0.2 0.2 0.8 1"},
    {"name": "torus_ring_green",  "radius": 0.05, "color": "0.2 0.8 0.2 1"},
    {"name": "torus_ring_orange", "radius": 0.04, "color": "0.8 0.5 0.2 1"},
    {"name": "torus_ring_pink",   "radius": 0.04, "color": "0.8 0.4 0.6 1"},
    {"name": "torus_ring_purple", "radius": 0.04, "color": "0.5 0.2 0.8 1"},
    {"name": "torus_ring_yellow", "radius": 0.04, "color": "0.8 0.8 0.2 1"},
]

for r in rings_config:
    create_ring_xml(r["name"], r["radius"], r["color"])

print("\nAll assets generation complete.")
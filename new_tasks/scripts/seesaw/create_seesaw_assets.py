import os

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

ASSET_ROOT = "custom_assets"
create_folder(ASSET_ROOT)

# ==========================================
# 1. Seesaw XML Generator
# ==========================================
def generate_seesaw_xml():
    # 尺寸定义
    base_h = 0.06
    board_l = 0.40
    board_w = 0.08
    board_h = 0.01
    fence_h = 0.015 # 围栏高度
    angle_range = 0.2
    
    # 围栏厚度
    t = 0.005
    
    # 托盘中心距离中心的偏移量
    offset = board_l / 2 - 0.04 # 离边缘4cm处
    
    # 生成围栏的 geom 字符串 (左右两端)
    fences = []
    # 左端围栏 (-offset)
    lx = -offset
    # 前后左右四片
    fences.append(f'<geom name="f1" type="box" size="{t} {board_w/2} {fence_h}" pos="{lx - 0.04} 0 {board_h+fence_h}" rgba="0.4 0.4 0.4 1" group="0"/>') # 左外
    fences.append(f'<geom name="f2" type="box" size="{t} {board_w/2} {fence_h}" pos="{lx + 0.04} 0 {board_h+fence_h}" rgba="0.4 0.4 0.4 1" group="0"/>') # 左内
    fences.append(f'<geom name="f3" type="box" size="{0.04} {t} {fence_h}" pos="{lx} {board_w/2} {board_h+fence_h}" rgba="0.4 0.4 0.4 1" group="0"/>') # 左上
    fences.append(f'<geom name="f4" type="box" size="{0.04} {t} {fence_h}" pos="{lx} {-board_w/2} {board_h+fence_h}" rgba="0.4 0.4 0.4 1" group="0"/>') # 左下
    
    # 右端围栏 (+offset)
    rx = offset
    fences.append(f'<geom name="f5" type="box" size="{t} {board_w/2} {fence_h}" pos="{rx + 0.04} 0 {board_h+fence_h}" rgba="0.4 0.4 0.4 1" group="0"/>') # 右外
    fences.append(f'<geom name="f6" type="box" size="{t} {board_w/2} {fence_h}" pos="{rx - 0.04} 0 {board_h+fence_h}" rgba="0.4 0.4 0.4 1" group="0"/>') # 右内
    fences.append(f'<geom name="f7" type="box" size="{0.04} {t} {fence_h}" pos="{rx} {board_w/2} {board_h+fence_h}" rgba="0.4 0.4 0.4 1" group="0"/>') # 右上
    fences.append(f'<geom name="f8" type="box" size="{0.04} {t} {fence_h}" pos="{rx} {-board_w/2} {board_h+fence_h}" rgba="0.4 0.4 0.4 1" group="0"/>') # 右下

    fence_str = "\n        ".join(fences)
    fence_str = ""

    xml_content = f"""
<mujoco model="seesaw">
  <asset>
    <texture name="tex_wood" type="2d" builtin="flat" rgb1="0.6 0.5 0.3" width="512" height="512"/>
    <material name="mat_wood" texture="tex_wood"/>
    <texture name="tex_base" type="2d" builtin="flat" rgb1="0.2 0.2 0.2" width="512" height="512"/>
    <material name="mat_base" texture="tex_base"/>
  </asset>
  <worldbody>
    <body>
      <body name="object">
        <!-- 1. 底座 (固定，重) -->
        <geom name="base_geom" type="box" size="0.05 0.1 {base_h/2}" pos="0 0 {base_h/2}" material="mat_base" 
              density="50000" friction="2.0 0.005 0.0001" group="0"/>
        <geom name="base_vis" type="box" size="0.05 0.1 {base_h/2}" pos="0 0 {base_h/2}" material="mat_base" 
              contype="0" conaffinity="0" group="1"/>

        <!-- 2. 跷跷板本体 (Hinge Joint) -->
        <!-- 支点高度在 base_h * 2 处，即 0.12m -->
        <body name="seesaw_board" pos="0 0 {base_h}">
            <!-- 关键：Hinge 关节，绕 Y 轴旋转，限制角度 +/- 25度 (约0.4弧度) -->
            <!-- damping=1.0 稍微大一点，防止晃个不停 -->
            <joint name="seesaw_joint" type="hinge" axis="0 1 0" range="-{angle_range} {angle_range}" damping="0.0" frictionloss="0.1" limited="true"/>
            
            <!-- 板子 -->
            <geom name="board_geom" type="box" size="{board_l/2} {board_w/2} {board_h}" material="mat_wood" 
                  density="800" friction="1.5 0.005 0.0001" group="0"/>
            <geom name="board_vis" type="box" size="{board_l/2} {board_w/2} {board_h}" material="mat_wood" 
                  contype="0" conaffinity="0" group="1"/>
            
            <!-- 围栏 -->
            {fence_str}

            <!-- Sites 用于定位 -->
            <!-- <site name="left_tray_site" pos="{lx} 0 {board_h}" size="0.01" rgba="1 0 0 0.5"/>
            <site name="right_tray_site" pos="{rx} 0 {board_h}" size="0.01" rgba="0 1 0 0.5"/> -->
        </body>
        
        <site name="bottom_site" pos="0 0 -0.01" size="0.002" rgba="0 0 0 0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""
    return xml_content

# ==========================================
# 2. Weights XML Generator
# ==========================================
def generate_weight_xml(name, size, color_rgb, density=1000):
    # size 是半长
    xml = f"""
<mujoco model="{name}">
  <worldbody>
    <body>
      <body name="object">
        <geom name="block_geom" type="box" size="{size} {size} {size}" rgba="{color_rgb} 1" 
              density="{density}" friction="1.5 0.005 0.0001" solref="0.001 1" solimp="0.95 0.99 0.001"
              condim="4" group="0"/>
        <geom name="block_vis" type="box" size="{size} {size} {size}" rgba="{color_rgb} 1" 
              contype="0" conaffinity="0" group="1"/>
        
        <site name="bottom_site" pos="0 0 {-size}" size="0.002" rgba="0 0 0 0"/>
        <site name="top_site" pos="0 0 {size}" size="0.002" rgba="0 0 0 0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""
    return xml

# ==========================================
# 3. Target Plate XML Generator
# ==========================================
def generate_plate_xml(name, size_half, height_half, color_rgb):
    xml = f"""
<mujoco model="{name}">
  <asset>
    <texture name="tex_plate" type="2d" builtin="flat" rgb1="{color_rgb}" width="512" height="512"/>
    <material name="mat_plate" texture="tex_plate"/>
  </asset>
  <worldbody>
    <body>
      <body name="object">
        <geom name="plate_geom" type="box" size="{size_half} {size_half} {height_half}" material="mat_plate" 
              density="5000" friction="1.0 0.005 0.0001" group="0"/>
        <geom name="plate_vis" type="box" size="{size_half} {size_half} {height_half}" material="mat_plate" 
              contype="0" conaffinity="0" group="1"/>
        
        <site name="bottom_site" pos="0 0 {-height_half}" size="0.002" rgba="0 0 0 0"/>
        <site name="top_site" pos="0 0 {height_half}" size="0.002" rgba="0 0 0 0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""
    return xml

# ==========================================
# Main Execution
# ==========================================
# 1. 生成跷跷板
seesaw_dir = os.path.join(ASSET_ROOT, "seesaw")
create_folder(seesaw_dir)
with open(os.path.join(seesaw_dir, "seesaw.xml"), "w") as f:
    f.write(generate_seesaw_xml())

# 2. 生成 3 个方块
weights_config = [
    ("weight_light", 0.02, "0.9 0.9 0.1", 500),   # 4cm, Yellow, Light
    ("weight_medium", 0.02, "0.0 1.0 1.0", 1000), # 4cm, Cyan, Medium
    ("weight_heavy", 0.02, "0.5 0.0 0.5", 3000),   # 4cm, Purple, Heavy
]

for name, size, rgb, density in weights_config:
    w_dir = os.path.join(ASSET_ROOT, name)
    create_folder(w_dir)
    with open(os.path.join(w_dir, f"{name}.xml"), "w") as f:
        f.write(generate_weight_xml(name, size, rgb, density))

# 3. 生成目标底座 (红色)
plate_name = "target_plate"
plate_dir = os.path.join(ASSET_ROOT, plate_name)
create_folder(plate_dir)
# 10cm x 10cm, 厚度 0.5cm
with open(os.path.join(plate_dir, f"{plate_name}.xml"), "w") as f:
    f.write(generate_plate_xml(plate_name, 0.05, 0.0025, "1 0 0"))

print("\n✅ Seesaw, Weights, and Plate Assets Generated.")
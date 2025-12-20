import os
import numpy as np

# ==========================================
# 1. 全局尺寸配置
# ==========================================
BALL_RADIUS = 0.02
RATIO = 1.3
CELL_SIZE = BALL_RADIUS * 2 * RATIO 
WALL_HEIGHT = BALL_RADIUS * 0.8

# 地板高度 (2mm)
# 因为我们不再分层，所以只需要定义一个统一的厚度
FLOOR_THICKNESS = 0.0020

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

ASSET_ROOT = "custom_assets"
create_folder(ASSET_ROOT)

def parse_maze_file(filename):
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found. Using default 15x15 layout.")
        return np.zeros((15,15)), (1,1), (13,13)
        
    with open(filename, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    
    start_pos = (1, 1)
    end_pos = (13, 13)
    matrix = []
    reading_layout = False

    for line in lines:
        if line.startswith("START:"):
            parts = line.split(":")[1].split(",")
            start_pos = (int(parts[0]), int(parts[1]))
        elif line.startswith("END:"):
            parts = line.split(":")[1].split(",")
            end_pos = (int(parts[0]), int(parts[1]))
        elif line.startswith("LAYOUT:"):
            reading_layout = True
            continue
        elif reading_layout:
            row = [int(x) for x in line.split()]
            matrix.append(row)

    return np.array(matrix), start_pos, end_pos

def grid_to_world(r, c, rows, cols, cell_size):
    pos_x = (c - (cols - 1) / 2.0) * cell_size
    pos_y = ((rows - 1) / 2.0 - r) * cell_size
    return pos_x, pos_y

# ==========================================
# 2. 生成迷宫本体 (Maze Structure)
# ==========================================
matrix, start_node, end_node = parse_maze_file("maze_def.txt")
rows, cols = matrix.shape

if rows == 0: rows = 15
if cols == 0: cols = 15

# 计算尺寸
wall_half_size = CELL_SIZE / 2.0
floor_cell_half = CELL_SIZE / 2.0 

geom_str = ""

for r in range(rows):
    for c in range(cols):
        wx, wy = grid_to_world(r, c, rows, cols, CELL_SIZE)
        
        # 默认地板颜色 (浅灰/白)
        rgba_str = "0.7 0.7 0.7 1" 
        
        # 特殊位置颜色
        if (r, c) == start_node:
            rgba_str = "0.8 0.2 0.2 1" # 起点红
        elif (r, c) == end_node:
            rgba_str = "0.2 0.8 0.2 1" # 终点绿

        # --- 修改部分 START ---
        
        # 1. 物理地板 (Group 1): 负责碰撞，不可见 (Robosuite不渲染Group 1)
        geom_str += f"""
        <geom name="floor_col_{r}_{c}" type="box" group="1" 
              size="{floor_cell_half:.4f} {floor_cell_half:.4f} {FLOOR_THICKNESS:.4f}" 
              pos="{wx:.4f} {wy:.4f} 0"
              rgba="{rgba_str}"
              condim="3" contype="1" conaffinity="1"
              friction="0.01 0.005 0.0001" solref="0.002 1"/>
        """

        # 2. 视觉地板 (Group 0): 负责颜色，无碰撞 (contype=0, conaffinity=0)
        # 注意：为了防止视觉上和桌子闪烁 (Z-fighting)，稍微把视觉层抬高极其微小的一点点 (例如 0.0001)，或者保持一致完全依赖 bottom_site
        # geom_str += f"""
        # <geom name="floor_vis_{r}_{c}" type="box" group="0" 
        #       size="{floor_cell_half:.4f} {floor_cell_half:.4f} {FLOOR_THICKNESS:.4f}" 
        #       rgba="{rgba_str}"
        #       pos="{wx:.4f} {wy:.4f} 0" 
        #       contype="0" conaffinity="0"/>
        # """
        # --- 修改部分 END ---

# --- 墙壁 (Walls) ---
for r in range(rows):
    for c in range(cols):
        if matrix[r][c] == 1:
            wx, wy = grid_to_world(r, c, rows, cols, CELL_SIZE)
            # 墙壁高度叠加在地板之上
            wall_z = FLOOR_THICKNESS + WALL_HEIGHT
            
            geom_str += f"""
        <geom name="wall_{r}_{c}" type="box" group="1" 
              size="{wall_half_size:.4f} {wall_half_size:.4f} {WALL_HEIGHT:.4f}" 
              material="mat_wall" condim="4"
              pos="{wx:.4f} {wy:.4f} {wall_z:.4f}"
              solref="0.002 1"/> 
        <geom name="wall_vis_{r}_{c}" type="box" group="0" 
              size="{wall_half_size:.4f} {wall_half_size:.4f} {WALL_HEIGHT:.4f}" 
              material="mat_wall" 
              pos="{wx:.4f} {wy:.4f} {wall_z:.4f}" contype="0" conaffinity="0"/>
            """

# 计算整个迷宫的底部Z坐标。因为地板中心在0，厚度是FLOOR_THICKNESS，所以底部是 -FLOOR_THICKNESS
bottom_z = -FLOOR_THICKNESS

maze_xml_content = f"""
<mujoco model="maze_structure">
  <asset>
    <texture name="tex_wall" type="2d" builtin="flat" rgb1="0.4 0.3 0.2" width="512" height="512"/>
    <material name="mat_wall" texture="tex_wall" shininess="0.1" specular="0.1"/>
  </asset>
  <worldbody>
    <body>
      <body name="object">
{geom_str}
        <site name="top_site" pos="0 0 0.05" size="0.002" rgba="0 0 0 0"/>
        <!-- 关键修复：添加 bottom_site -->
        <site name="bottom_site" pos="0 0 {bottom_z:.4f}" size="0.002" rgba="0 0 0 0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

maze_dir = os.path.join(ASSET_ROOT, "maze_structure")
create_folder(maze_dir)
with open(os.path.join(maze_dir, "maze_structure.xml"), "w") as f:
    f.write(maze_xml_content)


# ==========================================
# 3. 生成球 (Maze Ball)
# ==========================================
# 保持不变
ball_xml_content = f"""
<mujoco model="maze_ball">
  <asset>
    <texture name="tex_ball" type="2d" builtin="flat" rgb1="0 0 1" width="512" height="512"/>
    <material name="mat_ball" texture="tex_ball" shininess="0.3" specular="0.3"/>
  </asset>
  <worldbody>
    <body>
      <body name="object">
        <geom name="ball_geom" type="sphere" size="{BALL_RADIUS}" material="mat_ball" 
              density="1000" friction="0.005 0.0005 0.00001" solref="0.002 1" solimp="0.99 0.99 0.001"
              condim="4" group="1"/>
        <geom name="ball_vis" type="sphere" size="{BALL_RADIUS}" material="mat_ball" 
              contype="0" conaffinity="0" group="0"/>
        <site name="bottom_site" pos="0 0 -{BALL_RADIUS}" size="0.002" rgba="0 0 0 0"/>
        <site name="top_site" pos="0 0 {BALL_RADIUS}" size="0.002" rgba="0 0 0 0"/>
        <site name="horizontal_radius_site" pos="{BALL_RADIUS} 0 0" size="0.002" rgba="0 0 0 0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

ball_dir = os.path.join(ASSET_ROOT, "maze_ball")
create_folder(ball_dir)
with open(os.path.join(ball_dir, "maze_ball.xml"), "w") as f:
    f.write(ball_xml_content)

print(f"✅ Assets generated. Solution: Removed global invisible floor. Tiles are now both visual and physical.")
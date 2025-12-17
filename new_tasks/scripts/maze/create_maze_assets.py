import os
import numpy as np

# ==========================================
# 1. 全局尺寸配置 (修改比例为 1.2)
# ==========================================
BALL_RADIUS = 0.015
RATIO = 1.5  # [修改] 从 1.5 降为 1.2，迷宫更紧凑
CELL_SIZE = BALL_RADIUS * 2 * RATIO 
WALL_HEIGHT = BALL_RADIUS * RATIO 
FLOOR_THICKNESS = 0.002  # 地板半高

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

ASSET_ROOT = "custom_assets"
create_folder(ASSET_ROOT)

def parse_maze_file(filename):
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

floor_half_x = (cols * CELL_SIZE) / 2.0
floor_half_y = (rows * CELL_SIZE) / 2.0
wall_half_size = CELL_SIZE / 2.0

# 计算标记坐标
start_wx, start_wy = grid_to_world(start_node[0], start_node[1], rows, cols, CELL_SIZE)
end_wx, end_wy = grid_to_world(end_node[0], end_node[1], rows, cols, CELL_SIZE)

# [修改] 标记大小：填满格子的 80%
marker_half_size = wall_half_size * 0.8
# [修改] 标记高度：地板表面再抬高 1mm，防止 Z-fighting
marker_z = FLOOR_THICKNESS + 0.001 
# [修改] 标记厚度：设为 0.5mm，使其看起来像地毯
marker_thickness = 0.0005

geom_str = ""

# --- 地板 ---
geom_str += f"""
        <!-- 地板碰撞体 (隐形) -->
        <geom name="floor_col" type="box" group="1" 
              size="{floor_half_x:.4f} {floor_half_y:.4f} {FLOOR_THICKNESS:.4f}" 
              material="mat_maze" pos="0 0 0"
              friction="1.0 0.005 0.0001" solref="0.002 1"/>
        
        <!-- 地板视觉体 -->
        <geom name="floor_vis" type="box" group="0" 
              size="{floor_half_x:.4f} {floor_half_y:.4f} {FLOOR_THICKNESS:.4f}" 
              material="mat_maze" pos="0 0 0" contype="0" conaffinity="0"/>
"""

# --- [修改] 起终点标记 (方形 Box) ---
geom_str += f"""
        <!-- 起点标记 (红色方形) -->
        <geom name="start_marker" type="box" group="0"
              size="{marker_half_size:.4f} {marker_half_size:.4f} {marker_thickness:.4f}"
              rgba="0.8 0 0 0.8" 
              pos="{start_wx:.4f} {start_wy:.4f} {marker_z:.4f}"
              contype="0" conaffinity="0"/>

        <!-- 终点标记 (绿色方形) -->
        <geom name="end_marker" type="box" group="0"
              size="{marker_half_size:.4f} {marker_half_size:.4f} {marker_thickness:.4f}"
              rgba="0 0.8 0 0.8" 
              pos="{end_wx:.4f} {end_wy:.4f} {marker_z:.4f}"
              contype="0" conaffinity="0"/>
"""

# --- 墙壁 ---
for r in range(rows):
    for c in range(cols):
        if matrix[r][c] == 1:
            wx, wy = grid_to_world(r, c, rows, cols, CELL_SIZE)
            # 墙壁底部接在地板表面
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

maze_xml_content = f"""
<mujoco model="maze_structure">
  <asset>
    <texture name="tex_maze" type="2d" builtin="flat" rgb1="0.9 0.9 0.9" width="512" height="512"/>
    <material name="mat_maze" texture="tex_maze" shininess="0.1" specular="0.1"/>
    <texture name="tex_wall" type="cube" builtin="flat" rgb1="0.4 0.3 0.2" width="512" height="512"/>
    <material name="mat_wall" texture="tex_wall" shininess="0.1" specular="0.1"/>
  </asset>
  <worldbody>
    <body>
      <body name="object">
{geom_str}
        <site name="top_site" pos="0 0 0.05" size="0.002" rgba="0 0 0 0"/>
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
# 3. 生成球 (Maze Ball) - 保持不变
# ==========================================
ball_xml_content = f"""
<mujoco model="maze_ball">
  <asset>
    <texture name="tex_ball" type="cube" builtin="flat" rgb1="0 0 1" width="512" height="512"/>
    <material name="mat_ball" texture="tex_ball" shininess="0.8" specular="0.8"/>
  </asset>
  <worldbody>
    <body>
      <body name="object">
        <geom name="ball_geom" type="sphere" size="{BALL_RADIUS}" material="mat_ball" 
              density="1000" friction="0.5 0.005 0.0001" solref="0.002 1" solimp="0.99 0.99 0.001"
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

print(f"✅ Assets generated: Ratio 1.2, Square Markers added.")
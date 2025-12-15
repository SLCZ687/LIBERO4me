import numpy as np
import os
from collections import deque

# 全局配置，保持与 my_maze_task.py 一致
BALL_RADIUS = 0.015

def parse_maze_file(filename):
    """
    解析 maze_def.txt 文件
    格式要求：
    START: r, c
    END: r, c
    LAYOUT:
    0 1 ...
    """
    with open(filename, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    start_pos = None
    end_pos = None
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

def solve_maze_bfs(matrix, start_node, end_node):
    """
    使用 BFS 寻找最短路径 (网格坐标)
    """
    rows, cols = matrix.shape
    queue = deque([[start_node]])
    visited = set([start_node])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # 上下左右
    
    while queue:
        path = queue.popleft()
        r, c = path[-1]
        
        if (r, c) == end_node:
            return path
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and matrix[nr][nc] == 0 and (nr, nc) not in visited:
                queue.append(path + [(nr, nc)])
                visited.add((nr, nc))
    
    raise ValueError(f"无法找到从 {start_node} 到 {end_node} 的路径，请检查迷宫定义！")

def grid_to_world(r, c, rows, cols, cell_size):
    """将网格坐标转换为 MuJoCo 世界坐标"""
    # X轴: 列索引 c -> 对应 MuJoCo x
    pos_x = (c - (cols - 1) / 2.0) * cell_size
    # Y轴: 行索引 r -> 对应 MuJoCo y (反转)
    pos_y = ((rows - 1) / 2.0 - r) * cell_size
    return pos_x, pos_y

def generate_maze_files(def_file="maze_def.txt", xml_file="maze.xml", path_file="maze_path.npy"):
    # 1. 解析文件
    if not os.path.exists(def_file):
        raise FileNotFoundError(f"找不到迷宫定义文件: {def_file}")
    
    matrix, start_node, end_node = parse_maze_file(def_file)
    rows, cols = matrix.shape
    
    # 2. 计算尺寸
    # 单元格边长 (保留你之前的设定 1.05倍)
    cell_size = BALL_RADIUS * 2 * 1.05
    wall_half_size = cell_size / 2.0
    wall_height = BALL_RADIUS * 1.2
    floor_half_x = (cols * cell_size) / 2.0
    floor_half_y = (rows * cell_size) / 2.0

    # 3. 生成 XML 内容
    xml_content = f"""<mujoco model="generated_maze">
    <asset>
        <material name="wall_mat" rgba="0.8 0.5 0.3 1" shininess="0.1"/>
        <material name="floor_mat" rgba="0.7 0.4 0.2 1" shininess="0.0"/>
    </asset>
    
    <worldbody>
        <body name="root_wrapper" pos="0 0 0">
            <body name="object" pos="0 0 0">
                <geom name="floor" type="box" group="0" 
                      size="{floor_half_x:.4f} {floor_half_y:.4f} 0.005" 
                      material="floor_mat" pos="0 0 0"/>
"""
    
    wall_count = 0
    for r in range(rows):
        for c in range(cols):
            if matrix[r][c] == 1:
                pos_x, pos_y = grid_to_world(r, c, rows, cols, cell_size)
                xml_content += f"""                <geom name="wall_{r}_{c}" type="box" group="0" 
                      size="{wall_half_size:.4f} {wall_half_size:.4f} {wall_height:.4f}" 
                      material="wall_mat" 
                      pos="{pos_x:.4f} {pos_y:.4f} {wall_height:.4f}"/>\n"""
                wall_count += 1

    xml_content += """            </body>
        </body>
    </worldbody>
</mujoco>
"""
    
    # 保存 XML
    with open(xml_file, "w") as f:
        f.write(xml_content)
    print(f"[Generate] XML 已生成: {xml_file} (墙壁数: {wall_count})")

    # 4. 计算路径并保存
    grid_path = solve_maze_bfs(matrix, start_node, end_node)
    
    world_path = []
    for r, c in grid_path:
        wx, wy = grid_to_world(r, c, rows, cols, cell_size)
        # Z轴高度设为球心高度
        world_path.append([wx, wy, BALL_RADIUS])
    
    world_path = np.array(world_path)
    
    # 保存路径数据供任务读取
    np.save(path_file, world_path)
    print(f"[Generate] 路径已计算并保存: {path_file} (节点数: {len(world_path)})")

if __name__ == "__main__":
    generate_maze_files()
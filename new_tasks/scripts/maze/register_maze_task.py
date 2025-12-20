import os
import numpy as np
import libero.libero.envs.objects.custom_objects
from libero.libero.utils.mu_utils import register_mu, InitialSceneTemplates
from libero.libero.utils.task_generation_utils import register_task_info, generate_bddl_from_task_info
from libero.libero.utils.bddl_generation_utils import get_xy_region_kwargs_list_from_regions_info

# ==========================================
# 1. 参数同步
# ==========================================
BALL_RADIUS = 0.02
CELL_SIZE = BALL_RADIUS * 2 * 1.3

maze_offset_x = -0.20

def parse_maze_def(filename="maze_def.txt"):
    if not os.path.exists(filename): return np.zeros((15,15)), (1,1), (13,13)
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

@register_mu(scene_type="kitchen")
class MazeScene(InitialSceneTemplates):
    def __init__(self):
        fixture_num_info = {
            "kitchen_table": 1,
        }
        object_num_info = {
            "maze_structure": 1,
            "maze_ball": 1,
        }
        super().__init__(
            workspace_name="kitchen_table",
            fixture_num_info=fixture_num_info,
            object_num_info=object_num_info,
        )

    def get_region_dict(self, region_centroid_xy, region_name, target_name=None, region_half_len=0.02, yaw_rotation=(0.0, 0.0)):
        if isinstance(region_half_len, (list, tuple)):
            hx, hy = region_half_len
            if target_name is None: target_name = self.workspace_name
            return {
                region_name: {
                    "target": target_name,
                    "ranges": [(region_centroid_xy[0]-hx, region_centroid_xy[1]-hy, region_centroid_xy[0]+hx, region_centroid_xy[1]+hy)],
                    "yaw_rotation": [yaw_rotation],
                }
            }
        else:
            return super().get_region_dict(region_centroid_xy, region_name, target_name, region_half_len, yaw_rotation)

    def define_regions(self):
        matrix, start_node, end_node = parse_maze_def()
        rows, cols = matrix.shape
        
        start_x, start_y = grid_to_world(start_node[0], start_node[1], rows, cols, CELL_SIZE)
        end_x, end_y = grid_to_world(end_node[0], end_node[1], rows, cols, CELL_SIZE)
        
        # 迷宫中心
        self.regions.update(self.get_region_dict(
            region_centroid_xy=[maze_offset_x, 0.0],
            region_name="maze_center_region",
            target_name=self.workspace_name,
            region_half_len=0.005
        ))
        
        # Start Node
        self.regions.update(self.get_region_dict(
            region_centroid_xy=[start_x + maze_offset_x, start_y],
            region_name="ball_start_region",
            target_name=self.workspace_name,
            region_half_len=0.001 
        ))

        # End Node
        self.regions.update(self.get_region_dict(
            region_centroid_xy=[end_x + maze_offset_x, end_y],
            region_name="ball_target_region",
            target_name=self.workspace_name,
            region_half_len=0.02 
        ))
        
        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(self.regions)

    @property
    def init_states(self):
        states = [
            ("On", "maze_structure_1", "kitchen_table_maze_center_region"),
            ("On", "maze_ball_1", "kitchen_table_ball_start_region")
        ]
        return states

if __name__ == "__main__":
    scene_name = "maze_scene"
    language = "Push the ball to the goal"
    
    register_task_info(language,
                    scene_name=scene_name,
                    objects_of_interest=["maze_ball_1"],
                    goal_states=[
                        ("On", "maze_ball_1", "kitchen_table_ball_target_region")
                    ],
    )

    BDDL_FOLDER = "./custom_pddl"
    bddl_files, failures = generate_bddl_from_task_info(folder=BDDL_FOLDER)
    
    print(f"\nGenerated Task Files: {bddl_files}")
    if failures: print("Failures:", failures)
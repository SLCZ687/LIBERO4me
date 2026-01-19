import pathlib
# 集中配置文件路径
# 您可以在这里统一修改源目录名称和目标目录路径

# 当前目录下的两个源子目录名称
SOURCE_DIR_ASSETS = "custom_assets"
SOURCE_DIR_BDDL = "custom_pddl"

# 两个目标目录路径
project_root = pathlib.Path(__file__).resolve().parents[2]
print(f"Project Root: {project_root}")
TARGET_DIR_ASSETS = project_root / "libero/libero/assets/custom_objects"
TARGET_DIR_BDDL = project_root / "libero/libero/bddl_files/custom"
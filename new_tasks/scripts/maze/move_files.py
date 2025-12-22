import shutil
from pathlib import Path

# new_tasks/scripts/maze/move_files.py 往上 3 层到 LIBERO4me
REPO_ROOT = Path(__file__).resolve().parents[3]

# 目标目录：放自定义资产和自定义 bddl
TARGET_ASSETS_DIR = REPO_ROOT / "libero" / "libero" / "assets" / "custom_objects"
TARGET_BDDL_DIR = REPO_ROOT / "libero" / "libero" / "bddl_files" / "custom"

# 源目录：你项目根目录下的 custom_assets / custom_pddl
SOURCE_ASSETS_DIR = REPO_ROOT / "custom_assets"
SOURCE_BDDL_DIR = REPO_ROOT / "custom_pddl"


def move_directory_contents(source_dir: Path, target_dir: Path, overwrite: bool = True) -> None:
    """
    将源目录下的所有内容（文件和子目录）移动到目标目录

    Args:
        source_dir: 源目录路径 (Path)
        target_dir: 目标目录路径 (Path)
        overwrite: 是否覆盖重名文件/目录，默认True
    """
    source_path = source_dir.resolve()
    target_path = target_dir.resolve()

    if not source_path.exists():
        print(f"警告：源目录不存在，跳过：{source_path}")
        return

    target_path.mkdir(parents=True, exist_ok=True)

    for item in source_path.iterdir():
        target_item = target_path / item.name

        try:
            if target_item.exists() and overwrite:
                if target_item.is_dir():
                    shutil.rmtree(target_item)
                else:
                    target_item.unlink()

            shutil.move(str(item), str(target_path))
            print(f"成功移动: {item} -> {target_path}")

        except PermissionError:
            print(f"权限错误：无法移动 {item}，可能是文件被占用或权限不足")
        except Exception as e:
            print(f"移动失败 {item}：{e}")


def main():
    print("开始执行文件移动操作...")
    print("=" * 50)

    print(f"\n处理目录: {SOURCE_ASSETS_DIR}  ->  {TARGET_ASSETS_DIR}")
    move_directory_contents(SOURCE_ASSETS_DIR, TARGET_ASSETS_DIR)

    print(f"\n处理目录: {SOURCE_BDDL_DIR}  ->  {TARGET_BDDL_DIR}")
    move_directory_contents(SOURCE_BDDL_DIR, TARGET_BDDL_DIR)

    print("\n" + "=" * 50)
    print("文件移动操作执行完成！")


if __name__ == "__main__":
    confirm = input("即将执行文件移动操作，重名文件将被覆盖！是否继续？(y/n)：")
    if confirm.lower() == "y":
        main()
    else:
        print("操作已取消！")

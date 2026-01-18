import os
import shutil
import sys
from pathlib import Path

# 直接导入同级目录下的配置文件
try:
    from paths_config import SOURCE_DIR_ASSETS, SOURCE_DIR_BDDL, TARGET_DIR_ASSETS, TARGET_DIR_BDDL
except ImportError:
    # 尝试添加当前脚本所在目录到 path (防止某些特殊执行环境)
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from paths_config import SOURCE_DIR_ASSETS, SOURCE_DIR_BDDL, TARGET_DIR_ASSETS, TARGET_DIR_BDDL

def move_directory_contents(source_dir: str, target_dir: str, overwrite: bool = True) -> None:
    """
    将源目录下的所有内容（文件和子目录）移动到目标目录
    
    Args:
        source_dir: 源目录路径
        target_dir: 目标目录路径
        overwrite: 是否覆盖重名文件/目录，默认True
    """
    # 转换为Path对象，方便路径操作
    source_path = Path(source_dir).resolve()
    target_path = Path(target_dir).resolve()

    # 检查源目录是否存在
    if not source_path.exists():
        print(f"警告：源目录 {source_dir} 不存在，跳过该目录的移动操作")
        return

    # 创建目标目录（如果不存在）
    target_path.mkdir(parents=True, exist_ok=True)

    # 遍历源目录下的所有内容
    for item in source_path.iterdir():
        # 目标路径 = 目标目录 + 当前项的名称
        target_item = target_path / item.name

        try:
            # 如果目标项已存在且需要覆盖
            if target_item.exists() and overwrite:
                # 如果是目录，先删除目标目录（shutil.move无法直接覆盖非空目录）
                if target_item.is_dir():
                    shutil.rmtree(target_item)
                # 如果是文件，直接删除
                else:
                    target_item.unlink()

            # 移动文件/目录
            shutil.move(str(item), str(target_path))
            print(f"成功移动: {item} -> {target_path}")

        except PermissionError:
            print(f"权限错误：无法移动 {item}，可能是文件被占用或权限不足")
        except Exception as e:
            print(f"移动失败 {item}：{str(e)}")

def main():
    """主函数：配置源目录和目标目录并执行移动操作"""
    # ===================== 配置区域 =====================
    # 使用 paths_config.py 中的统一配置
    source_dir1 = SOURCE_DIR_ASSETS
    source_dir2 = SOURCE_DIR_BDDL
    
    target_dir1 = TARGET_DIR_ASSETS
    target_dir2 = TARGET_DIR_BDDL
    # ====================================================

    # 获取当前工作目录
    current_dir = Path.cwd()
    
    # 拼接完整路径
    full_source1 = current_dir / source_dir1
    full_source2 = current_dir / source_dir2
    # target_dir1 和 target_dir2 已经是绝对路径了，直接使用 Path 转换
    full_target1 = Path(target_dir1)
    full_target2 = Path(target_dir2)

    print("开始执行文件移动操作...")
    print("=" * 50)

    # 移动第一个目录的内容
    print(f"\n处理目录: {full_source1}")
    move_directory_contents(full_source1, full_target1)

    # 移动第二个目录的内容
    print(f"\n处理目录: {full_source2}")
    move_directory_contents(full_source2, full_target2)

    print("\n" + "=" * 50)
    print("文件移动操作执行完成！")

if __name__ == "__main__":
    # 确认操作提示（可选）
    confirm = input("即将执行文件移动操作，重名文件将被覆盖！是否继续？(y/n)：")
    if confirm.lower() == "y":
        main()
    else:
        print("操作已取消！")

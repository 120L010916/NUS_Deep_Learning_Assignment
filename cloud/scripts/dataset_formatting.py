from typing import List, Tuple
import os
from pathlib import Path
import argparse
import random

def build_argparser() -> argparse.ArgumentParser:
    """
    Build the argument parser for command line arguments.
    Returns:
        argparse.ArgumentParser: The argument parser.
    """
    parser = argparse.ArgumentParser(description="Process real clouds dataset.")
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Directory containing the images to process."
    )
    parser.add_argument(
        "--A", type=str, required=True,
        help="Directory containing the images of modality A to process."
    )
    parser.add_argument(
        "--B", type=str, required=True,
        help="Directory containing the images of modality B to process."
    )
    parser.add_argument(
        "--folders", type=str, nargs=4, required=True,
        help="List of folder names to create for training and testing data."
    )
    parser.add_argument(
        "--alpha", type=float, required=True,
        help="Ratio of training data to total data."
    )
    return parser


def browse_folder(
        path: Path, 
        A: str, 
        B: str) -> Tuple[List[Path], List[Path]]:
    """
    Browse a directory and return a list of all the png files in it and its subfolder.
    """
    # 支持 .png 和 .jpg 文件（任务要求 PNG，但之前保存为 JPG）
    valid_extensions = (".png", ".jpg", ".jpeg")
    
    # 转换为 Path 对象
    path_A = Path(path) / A
    path_B = Path(path) / B
    
    # 遍历 modality A 的文件
    files_A = []
    if path_A.exists():
        for root, _, files in os.walk(path_A):
            files_A.extend(
                Path(root) / f for f in files if f.lower().endswith(valid_extensions)
            )
    
    # 遍历 modality B 的文件
    files_B = []
    if path_B.exists():
        for root, _, files in os.walk(path_B):
            files_B.extend(
                Path(root) / f for f in files if f.lower().endswith(valid_extensions)
            )
    
    # 按文件名排序，确保一致性
    files_A.sort()
    files_B.sort()
    
    return files_A, files_B


def check_existence(
        filenames: List[Path]) -> None:
    """
    Check if the paths exist.
    Args:
        filenames (List[Path]): List of filenames.
    Returns:
        None
    """
    for filepath in filenames:
        if not filepath.exists():
            raise FileNotFoundError(f"File does not exist: {filepath}")


def create_folders(
        input_dir: Path, 
        folder_list: List[str]) -> None:
    """
    Create folders for training and testing data.
    Args:
        input_dir (Path): Input directory.
        folder_list (List[str]): List of folder names to create.
    Returns:
        None
    """
    for folder_name in folder_list:
        folder_path = input_dir / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)


def split_train_test(
        filenames: List[Path], 
        alpha: float) -> Tuple[List[Path], List[Path]]:
    """
    Split the filenames into training and testing sets.
    Args:
        filenames (List[Path]): List of filenames.
        alpha (float): Ratio of training data to total data.
    Returns:
        Tuple[List[Path], List[Path]]: Training and testing filenames.
    """
    # 验证 alpha
    if not 0 < alpha < 1:
        raise ValueError(f"Alpha must be between 0 and 1, got {alpha}.")
    
    # 随机打乱文件列表
    filenames_shuffled = filenames.copy()
    random.shuffle(filenames_shuffled)
    
    # 计算训练集大小
    train_size = int(len(filenames_shuffled) * alpha)
    
    # 划分训练和测试集
    train_files = filenames_shuffled[:train_size]
    test_files = filenames_shuffled[train_size:]
    
    return train_files, test_files


def create_symlinks(
        filenames: List[Path],
        input_dir: Path,
        split_dir: Path) -> None:
    """
    Create symbolic links for the training and testing images.
    Args:
        filenames (List[Path]): List of filenames.
        input_dir (Path): Input directory.
        split_dir (Path): Split directory.
    Returns:
        None
    """
    for filepath in filenames:
        # 目标链接路径（保持原文件名）
        link_path = split_dir / filepath.name
        
        # 如果链接已存在，先删除
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        
        # 使用绝对路径创建符号链接
        abs_filepath = filepath.resolve()  # 获取绝对路径
        try:
            link_path.symlink_to(abs_filepath)
        except OSError as e:
            print(f"Failed to create symlink for {abs_filepath}: {e}")


def process(input_dir: str, A: str, B: str, folders: List[str], alpha: float) -> None:
    """
    Format the dataset for training and testing.
    Args:
        data_dir (str): Directory containing the images.
        A (str): Name of the first dataset.
        B (str): Name of the second dataset.
        folders (List[str]): List of folder names to create.
        alpha (float): Ratio of training data to total data.
    Returns:
        None
    """
    # 转换为 Path 对象
    input_dir_path = Path(input_dir)
    
    # 创建子文件夹（trainA, trainB, testA, testB）
    create_folders(input_dir_path, folders)
    
    # 获取 modality A 和 B 的文件列表
    files_A, files_B = browse_folder(input_dir_path.parent, A, B)
    
    # 检查文件是否存在
    check_existence(files_A)
    check_existence(files_B)
    
    # 划分训练和测试集
    train_A, test_A = split_train_test(files_A, alpha)
    train_B, test_B = split_train_test(files_B, alpha)
    
    # 创建符号链接
    create_symlinks(train_A, input_dir_path, input_dir_path / folders[0])  # trainA
    create_symlinks(train_B, input_dir_path, input_dir_path / folders[1])  # trainB
    create_symlinks(test_A, input_dir_path, input_dir_path / folders[2])   # testA
    create_symlinks(test_B, input_dir_path, input_dir_path / folders[3])   # testB
    
    print(f"Dataset formatted successfully in {input_dir_path}")
    print(f"trainA: {len(train_A)} files, trainB: {len(train_B)} files")
    print(f"testA: {len(test_A)} files, testB: {len(test_B)} files")


def main():
    """
    Main function to execute the script.
    """
    parser = build_argparser()
    args = parser.parse_args()

    process(args.input_dir, args.A, args.B, args.folders, args.alpha)


if __name__ == "__main__":
    main()

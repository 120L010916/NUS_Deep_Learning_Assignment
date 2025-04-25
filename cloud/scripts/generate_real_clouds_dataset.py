import os
from typing import List, Tuple
from tqdm import tqdm
import argparse
from utils import load_image, patchify, save_patches, get_filenames


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
        "--crop", type=int, nargs=4, required=True,
        help="Coordinates for cropping (left, upper, right, lower)."
    )
    parser.add_argument(
        "--patch_size", type=int, nargs=2, required=True,
        help="Size of the patches (height, width)."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save the patches."
    )
    parser.add_argument(
        "--max_files", type=int, default=-1,
        help="Maximum number of files to process. Default is -1 (all files)."
    )
    return parser


def process(
        filenames: List[str], 
        patch_size: Tuple[int, int],
        crop: Tuple[int, int, int, int],
        output_dir: str) -> None:
    """
    Process the real clouds by loading the images, patchifying them,
    and saving the patches to the output directory.
    Args:
        filenames (List[str]): List of image filenames.
        patch_size (Tuple[int, int]): Size of the patches (height, width).
        crop (Tuple[int, int, int, int]): Coordinates for cropping (left, upper, right, lower).
        output_dir (str): Directory to save the patches.
    Returns:
        None
    """
    # 验证patch尺寸
    if patch_size != [256, 256]:
        raise ValueError("Patch size must be (256, 256) as per assignment requirements.")
    
    # 验证裁剪尺寸
    crop_width = crop[2] - crop[0]
    crop_height = crop[3] - crop[1]
    if crop_width != 1024 or crop_height != 512:
        raise ValueError("Crop dimensions must result in 1024x512 image.")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 全局patch索引
    global_patch_index = 0
    
    # 遍历所有文件名
    for filename in tqdm(filenames, desc="Processing images"):
        # 加载图片并应用裁剪
        img = load_image(filename, crop=crop)
        if img is None:
            print(f"Skipping {filename} due to load error.")
            continue
        
        # 验证裁剪后尺寸
        if img.shape[:2] != (512, 1024):
            print(f"Warning: {filename} has unexpected size {img.shape[:2]}, expected (512, 1024).")
            continue
        
        # 分解为8个256x256的patch
        try:
            patches = patchify(img, patch_size)
            if len(patches) != 8:
                print(f"Warning: Expected 8 patches for {filename}, got {len(patches)}.")
                continue
        except Exception as e:
            print(f"Error patchifying {filename}: {e}")
            continue
        
        # 保存patch
        try:
            save_patches(patches, output_dir, global_patch_index)
            global_patch_index += len(patches)  # 更新索引
        except Exception as e:
            print(f"Error saving patches for {filename}: {e}")


def main():
    """
    Main function to execute the script.
    """
    parser = build_argparser()
    args = parser.parse_args()
    filenames = get_filenames(args.input_dir)
    if args.max_files > 0:
        filenames = filenames[:args.max_files]
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    process(filenames, args.patch_size, args.crop, args.output_dir)


if __name__ == "__main__":
    main()

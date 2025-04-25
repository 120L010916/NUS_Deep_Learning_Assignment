import os
from typing import List, Tuple
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
from utils import load_image, patchify, save_patches, get_filenames
try:
    from noise import pnoise2
except ImportError:
    raise ImportError("Please install the 'noise' package: pip install noise")

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
        "--input_mask", type=str, required=True,
        help="Directory containing the mask."
    )
    parser.add_argument(
        "--patch_size", type=int, nargs=2, required=True,
        help="Size of the patches (height, width)."
    )
    parser.add_argument(
        "--crop", type=int, nargs=4, required=True,
        help="Coordinates for cropping (left, upper, right, lower)."
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


def generate_synthetic_clouds(
        shape: Tuple[int, int],
        res: Tuple[int, int],
        octaves: int) -> np.ndarray:
    """Generate a 2D numpy array of noise.
    Args:
        shape: The shape of the generated array (tuple of two ints).
            This must be a multiple of res.
        res: The number of periods of noise to generate along each
            axis (tuple of two ints). Note shape must be a multiple of
            res.
        octaves: The number of octaves in the noise.
    Returns:
        A numpy array of shape shape with the generated noise.
    """
    height, width = shape
    res_y, res_x = res
    
    # 验证 shape 是否为 res 的倍数
    if height % res_y != 0 or width % res_x != 0:
        raise ValueError(f"Shape {shape} must be a multiple of res {res}.")
    
    # 初始化噪声数组
    noise = np.zeros(shape, dtype=np.float32)
    
    # 生成 Perlin 噪声
    for y in range(height):
        for x in range(width):
            # 归一化坐标到噪声空间
            noise[y, x] = pnoise2(
                x / res_x, y / res_y,
                octaves=octaves,
                persistence=0.5,  # 默认值，控制振幅
                lacunarity=2.0,   # 默认值，控制频率
                repeatx=width // res_x,
                repeaty=height // res_y,
                base=0  # 随机种子
            )
    
    # 归一化噪声到 [0, 1]
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    
    return noise


def apply_synthetic_clouds_to_mask(
        noise: np.ndarray,
        mask: np.ndarray) -> np.ndarray:
    """Apply the generated noise to the mask. First, normalize the
    noise to be between 0 and 255, then add the noise to the mask.
    The mask is assumed to be in the range [0, 255]. Clip the output
    to be in the range [0, 255] and convert it to uint8.
    Args:
        noise: The generated noise.
        mask: The mask to apply the noise to.
    Returns:
        A numpy array of the same shape as the mask with the noise
        applied.
    """
    # 验证输入形状
    if noise.shape != mask.shape:
        raise ValueError(f"Noise shape {noise.shape} does not match mask shape {mask.shape}.")
    
    # 归一化噪声到 [0, 255]
    noise_normalized = (noise - noise.min()) / (noise.max() - noise.min()) * 255.0
    
    # 将噪声与 mask 结合
    # mask 是二值图像（0 或 255），噪声填充非边界区域
    result = np.where(mask == 255, mask, noise_normalized)
    
    # 裁剪到 [0, 255] 并转换为 uint8
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result


def process(
        N: int, 
        input_mask: str,
        patch_size: Tuple[int, int],
        crop: Tuple[int, int, int, int],
        output_dir: str) -> None:
    """
    Process the real clouds by loading the images, patchifying them,
    and saving the patches to the output directory.
    Args:
        N (int): Number of images to process.
        input_mask (str): Path to the input mask.
        patch_size (Tuple[int, int]): Size of the patches (height, width).
        crop (Tuple[int, int, int, int]): Coordinates for cropping (left, upper, right, lower).
        output_dir (str): Directory to save the patches.
    Returns:
        None
    """
    # 验证 patch 尺寸
    if patch_size != [256, 256]:
        raise ValueError("Patch size must be (256, 256) as per assignment requirements.")
    
    # 验证裁剪尺寸
    crop_width = crop[2] - crop[0]
    crop_height = crop[3] - crop[1]
    if crop_width != 1024 or crop_height != 512:
        raise ValueError("Crop dimensions must result in 1024x512 image.")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载 mask
    mask = load_image(input_mask, convert="L")  # 加载为灰度图
    if mask is None:
        raise ValueError(f"Failed to load mask from {input_mask}.")
    
    # 应用裁剪到 mask
    mask = load_image(input_mask, crop=crop, convert="L")
    if mask.shape[:2] != (512, 1024):
        raise ValueError(f"Mask after cropping has unexpected size {mask.shape[:2]}, expected (512, 1024).")
    
    # 全局 patch 索引
    global_patch_index = 0
    
    # 生成 N 张合成云图
    for _ in tqdm(range(N), desc="Generating synthetic clouds"):
        # 生成 Perlin 噪声（大小与裁剪后图片一致）
        noise = generate_synthetic_clouds(
            shape=(512, 1024),
            res=(64, 128),  # 噪声周期，调整以控制云纹理
            octaves=6       # 多层噪声，增加细节
        )
        
        # 应用噪声到 mask
        synthetic_image = apply_synthetic_clouds_to_mask(noise, mask)
        
        # 分解为 8 个 256x256 的 patch
        try:
            patches = patchify(synthetic_image, patch_size)
            if len(patches) != 8:
                print(f"Warning: Expected 8 patches, got {len(patches)}.")
                continue
        except Exception as e:
            print(f"Error patchifying synthetic image: {e}")
            continue
        
        # 保存 patches
        try:
            save_patches(patches, output_dir, global_patch_index)
            global_patch_index += len(patches)  # 更新索引
        except Exception as e:
            print(f"Error saving patches: {e}")


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
    process(len(filenames), args.input_mask, args.patch_size, args.crop, args.output_dir)


if __name__ == "__main__":
    main()

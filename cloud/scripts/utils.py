import os
from typing import List, Tuple, Union
from PIL import Image
import numpy as np


def get_filenames(input_dir: str) -> List[str]:
    """
    Get the list of image filenames from the input directory.
    Args:
        input_dir (str): Directory containing the images.
    Returns:
        List[str]: List of image filenames.
    """
    if not os.path.isdir(input_dir):
        raise ValueError(f"Input directory {input_dir} does not exist.")
    
    # 支持常见的图片格式
    valid_extensions = ('.jpg', '.jpeg', '.png')
    filenames = [
        os.path.join(input_dir, f) for f in os.listdir(input_dir)
        if f.lower().endswith(valid_extensions)
    ]
    
    # 按文件名排序以确保一致性
    filenames.sort()
    return filenames


def load_image(
        path: str,
        crop: Tuple[int, int, int, int] = None,
        convert: str = "RGB") -> Union[np.ndarray, None]:
    """
    Load an image from the given path and convert it to a numpy array.
    Args:
        path (str): Path to the image.
        crop (Tuple[int, int, int, int]): Coordinates for cropping (left, upper, right, lower).
        convert (str): Color mode to convert the image to (default is "RGB").
    Returns:
        np.ndarray: Numpy array of the image.
    """
    try:
        # 加载图片
        img = Image.open(path).convert(convert)
        
        # 应用裁剪（如果提供）
        if crop is not None:
            img = img.crop(crop)
        
        # 转换为NumPy数组
        img_array = np.array(img)
        return img_array
    
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return None



def patchify(
        img: np.ndarray,
        patch_size: Tuple[int, int] = (256, 256)) -> List[np.ndarray]:
    """
    Patchify the image into patches of size patch_sizes.
    Args:
        img (np.ndarray): The image to patchify.
        patch_sizes (Tuple[int, int]): The size of the patches.
    Returns:
        List[np.ndarray]: A list of patches.
    """
    if img is None:
        return []
    
    # 验证输入图片尺寸
    height, width = img.shape[:2]
    patch_height, patch_width = patch_size
    
    if height % patch_height != 0 or width % patch_width != 0:
        raise ValueError(f"Image size {width}x{height} is not divisible by patch size {patch_width}x{patch_height}.")
    
    # 计算patch数量
    num_patches_x = width // patch_width
    num_patches_y = height // patch_height
    
    patches = []
    for i in range(num_patches_y):
        for j in range(num_patches_x):
            # 提取patch
            patch = img[i * patch_height:(i + 1) * patch_height, j * patch_width:(j + 1) * patch_width]
            patches.append(patch)
    
    return patches


def save_patches(
        patches: List[np.ndarray], 
        output_dir: str,
        starting_index: int) -> None:
    """Save the synthetic clouds to the output directory.
    Args:
        patches: The list of patches.
        output_dir: The output directory.
        starting_index: The starting index for naming the files.
    Returns:
        None
    """
    if not patches:
        return
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存每个patch
    for idx, patch in enumerate(patches):
        try:
            # 转换为PIL Image
            patch_img = Image.fromarray(patch)
            # 生成文件名
            patch_name = f"patch_{starting_index + idx}.jpg"
            patch_path = os.path.join(output_dir, patch_name)
            # 保存为JPEG
            patch_img.save(patch_path, 'JPEG')
        except Exception as e:
            print(f"Error saving patch {patch_name}: {e}")

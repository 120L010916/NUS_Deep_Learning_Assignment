import numpy as np
from PIL import Image
import os

def generate_mask_from_image(image_path: str, output_path: str, crop: tuple = (21, 144, 1045, 656)) -> None:
    """
    Generate a binary mask from the input image by extracting green borders.
    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the output mask.
        crop (tuple): Crop coordinates (left, upper, right, lower).
    Returns:
        None
    """
    # 加载图片
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Failed to load image {image_path}: {e}")

    # 裁剪图片
    img = img.crop(crop)
    img_array = np.array(img)

    # 验证裁剪后尺寸
    if img_array.shape[:2] != (512, 1024):
        raise ValueError(f"Cropped image size {img_array.shape[:2]} does not match expected (512, 1024).")

    # 提取绿色边界
    # 假设绿色边界接近 RGB (0, 255, 0)，设置颜色范围
    lower_green = np.array([0, 150, 0])  # 绿色下界
    upper_green = np.array([100, 255, 100])  # 绿色上界

    # 创建二值 mask
    mask = np.zeros((512, 1024), dtype=np.uint8)
    green_pixels = np.all((img_array >= lower_green) & (img_array <= upper_green), axis=2)
    mask[green_pixels] = 255  # 绿色边界设为 255

    # 保存 mask
    mask_img = Image.fromarray(mask, mode="L")  # 灰度模式
    mask_img.save(output_path, "JPEG")
    print(f"Mask saved to {output_path}")

def main():
    # 输入和输出路径
    image_path = "/home/xulei/xulei/dataset-distillation/EE4115/assign2/cloud/collected_images/BlueMarbleASEAN_20250301_0000.jpg"
    output_path = "/home/xulei/xulei/dataset-distillation/EE4115/assign2/cloud/mask/input_mask.jpg"
    
    # 裁剪参数（与 generate_real_clouds_dataset.py 一致）
    crop = (21, 144, 1045, 656)
    
    # 生成 mask
    generate_mask_from_image(image_path, output_path, crop)

if __name__ == "__main__":
    main()
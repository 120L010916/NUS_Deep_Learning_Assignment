import os
import requests
from datetime import datetime, timedelta
import argparse
from pathlib import Path
import time

def build_argparser() -> argparse.ArgumentParser:
    """
    Build the argument parser for command line arguments.
    Returns:
        argparse.ArgumentParser: The argument parser.
    """
    parser = argparse.ArgumentParser(description="Process real clouds dataset.")
    parser.add_argument(
        "--start_date", type=str, required=True,
        help="Start date for downloading images (format: YYYY-MM-DD)."
    )
    parser.add_argument(
        "--end_date", type=str, required=True,
        help="End date for downloading images (format: YYYY-MM-DD)."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save the images."
    )
    return parser


def download_images(
        start_date: str, 
        end_date: str, 
        output_dir: str) -> None:
    """
    Downloads satellite images from NEA from start_date to end_date (inclusive).
    
    Args:
        start_date (str): Format 'YYYY-MM-DD'
        end_date (str): Format 'YYYY-MM-DD'
        output_dir (str): Directory to save images
    Returns:
        None
    """
    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 解析日期
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    # 初始化计数器
    total_images = 0
    target_images = 2500
    
    # 新的URL模板
    base_url = "https://www.nea.gov.sg/docs/default-source/satelliteimage/BlueMarbleASEAN_{}.jpg"
    
    current_date = start
    while current_date <= end and total_images < target_images:
        # 每20分钟一张图片，一天最多72张（24小时 * 60分钟 / 20分钟）
        for hour in range(0, 24):
            for minute in [0, 20, 40]:
                # 构造时间戳
                timestamp = current_date.replace(hour=hour, minute=minute)
                # 格式化时间戳为URL所需格式：YYYYMMDD_HHMM
                timestamp_str = timestamp.strftime("%Y%m%d_%H%M")
                
                # 构造图片URL
                image_url = base_url.format(timestamp_str)
                # 构造保存路径
                image_name = f"BlueMarbleASEAN_{timestamp_str}.jpg"
                # print(image_name)
                image_path = os.path.join(output_dir, image_name)
                
                # 如果文件已存在，跳过下载
                if os.path.exists(image_path):
                    print(f"Image {image_name} already exists, skipping...")
                    total_images += 1
                    continue
                
                try:
                    # 发送请求
                    response = requests.get(image_url, timeout=10)
                    
                    # 检查请求是否成功
                    if response.status_code == 200:
                        # 保存图片
                        with open(image_path, 'wb') as f:
                            f.write(response.content)
                        print(f"Downloaded {image_name}")
                        total_images += 1
                    else:
                        print(f"Failed to download {image_name}, status code: {response.status_code}")
                
                except requests.RequestException as e:
                    print(f"Error downloading {image_name}: {e}")
                
                # 控制请求频率，避免服务器限制
                time.sleep(0.5)
                
                # 检查是否达到目标数量
                if total_images >= target_images:
                    print(f"Reached target of {target_images} images.")
                    return
        
        # 移动到下一天
        current_date += timedelta(days=1)
    
    print(f"Download complete. Total images downloaded: {total_images}")


def main():
    """
    Main function to execute the script.
    """
    parser = build_argparser()
    args = parser.parse_args()
    download_images(args.start_date, args.end_date, args.output_dir)


if __name__ == "__main__":
   main()
    
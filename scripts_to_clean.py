import os
import shutil
from PIL import Image

# 定义源文件夹和目标文件夹
image_folder = './test/images'
json_folder = './test/jsons'
destination_ann_folder = './1044_1044/jsons'

destination_img_folder = './1044_1044/images'

# 确保目标文件夹存在
os.makedirs(destination_ann_folder, exist_ok=True)
os.makedirs(destination_img_folder, exist_ok=True)

# 遍历图像文件夹中的所有图像文件
for filename in os.listdir(image_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  # 根据需要添加其他图像格式
        # 构建图像的完整路径
        image_path = os.path.join(image_folder, filename)
        
        # 打开图像并获取尺寸
        with Image.open(image_path) as img:
            width, height = img.size
            
            # 检查图像尺寸是否为 (1353, 1020)
            if (width, height) == (1044, 1044):
                # 复制图像到目标文件夹
                shutil.copy(image_path, destination_img_folder)
                shutil.copy(image_path, destination_ann_folder)
                print(f'Copied image: {filename}')
                
                # 复制对应的 JSON 文件
                json_filename = os.path.splitext(filename)[0] + '.json'  # 假设 JSON 文件与图像同名
                json_path = os.path.join(json_folder, json_filename)
                if os.path.exists(json_path):
                    shutil.copy(json_path, destination_ann_folder)
                    print(f'Copied JSON: {json_filename}')
                else:
                    print(f'JSON file not found: {json_filename}')

print('所有符合条件的图像和对应的 JSON 文件已成功复制。')
import os
from PIL import Image
from collections import defaultdict

# 定义图像文件夹路径
image_folder = './test/images'

# 创建一个字典来存储尺寸分布
size_distribution = defaultdict(int)

# 遍历文件夹中的所有图像文件
for filename in os.listdir(image_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  # 根据需要添加其他图像格式
        # 构建图像的完整路径
        image_path = os.path.join(image_folder, filename)
        
        # 打开图像并获取尺寸
        with Image.open(image_path) as img:
            width, height = img.size
            size_distribution[(width, height)] += 1  # 记录尺寸出现次数

# 打印尺寸分布
print("图像尺寸分布:")
for size, count in size_distribution.items():
    print(f"尺寸 {size}: {count} 张图像")
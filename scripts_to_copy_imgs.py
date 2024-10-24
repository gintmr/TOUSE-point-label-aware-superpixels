import os
import shutil

# 定义源文件夹和目标文件夹
source_folder = './test/images'
destination_folder = './test/jsons'

# 确保目标文件夹存在
os.makedirs(destination_folder, exist_ok=True)

# 遍历源文件夹中的所有文件
for filename in os.listdir(source_folder):
    # 构建源文件的完整路径
    source_file = os.path.join(source_folder, filename)
    
    # 检查是否是文件（而不是文件夹）
    if os.path.isfile(source_file):
        # 复制文件到目标文件夹
        shutil.copy(source_file, destination_folder)
        print(f'Copied: {filename}')

print('所有图像已成功复制到 jsons 文件夹。')
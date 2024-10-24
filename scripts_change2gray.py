from PIL import Image
import numpy as np

# 打开原始图像
original_image = Image.open("./try/instance/row.png")

# 将图像转换为 RGB（如果不是的话）
original_image = original_image.convert("RGB")

# 将图像转换为灰度图像
gray_image = original_image.convert("L")

# 获取原始图像的像素数据
original_array = np.array(original_image)
gray_array = np.array(gray_image)

print("original_array shape:", original_array.shape)
gray_image.save("./try/instance/row_gray.png")
print("shape:", gray_array.shape)
# # 根据原始颜色修改灰度值
# # 这里我们可以使用原始图像的 R、G、B 通道的平均值来修改灰度值
# modified_gray_array = (original_array[..., 0] + original_array[..., 1] + original_array[..., 2]) / 3

# # 创建新的灰度图像
# modified_gray_image = Image.fromarray(modified_gray_array.astype(np.uint8), mode='L')

# # 保存修改后的灰度图像
# modified_gray_image.save("path/to/save/modified_gray_image.jpg")
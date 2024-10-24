import numpy as np

# 创建一个四维数组作为示例
B, H, W, C = 2, 3, 4, 5
labels_array = np.arange(B * H * W * C).reshape((B, H, W, C))

# 选择第二维的所有元素
selected_slice = labels_array[:, 1]

print("Original array shape:", labels_array.shape)
print("Original array:", labels_array)
print("Selected slice shape:", selected_slice.shape)
print("Selected slice:", selected_slice)
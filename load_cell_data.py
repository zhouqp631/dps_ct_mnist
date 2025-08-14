import torch
import numpy as np
import matplotlib.pyplot as plt


# def crop_around_max_no_pad(image, crop_size=64):
#     """
#     以最大值位置为中心附近裁剪固定大小的区域，不填充，自动调整位置防止越界。
#
#     参数:
#         image: 输入的2D张量（H × W）
#         crop_size: 裁剪的目标尺寸（默认64×64）
#
#     返回:
#         cropped_image: 裁剪后的图像（crop_size × crop_size）
#     """
#     assert len(image.shape) == 2, "输入应为2D张量（H × W）"
#     h, w = image.shape
#     half_size = crop_size // 2
#
#     # 找到最大值位置（取第一个）
#     max_pos = (image == torch.max(image)).nonzero()[0]
#     max_y, max_x = max_pos[0].item(), max_pos[1].item()
#
#     # 初始裁剪区域（可能越界）
#     y_start = max_y - half_size
#     y_end = max_y + half_size
#     x_start = max_x - half_size
#     x_end = max_x + half_size
#
#     # 调整越界的裁剪区域
#     if y_start < 0:
#         y_start = 0
#         y_end = crop_size
#     if y_end > h:
#         y_end = h
#         y_start = h - crop_size
#     if x_start < 0:
#         x_start = 0
#         x_end = crop_size
#     if x_end > w:
#         x_end = w
#         x_start = w - crop_size
#
#     # 执行裁剪
#     cropped = image[y_start:y_end, x_start:x_end]
#
#     # 检查是否成功裁剪到目标尺寸
#     assert cropped.shape == (crop_size, crop_size), \
#         f"裁剪失败，得到 {cropped.shape}，但期望 {crop_size}x{crop_size}"
#
#     return cropped
#
#
# # 加载原始数据
# 加载裁剪后的数据
cropped_data = np.load('cropped_inverse_scatter.npz')['arr_0']
print(f"数据形状: {cropped_data.shape} (N × H × W)")

# 随机选择n张图片可视化00000
n = 4  # 查看4张
random_indices = np.random.choice(len(cropped_data), size=n, replace=False)

# 创建画布
plt.figure(figsize=(15, 10))
for i, idx in enumerate(random_indices, 1):
    plt.subplot(2, 2, i)
    plt.imshow(cropped_data[idx])
    plt.colorbar()
    plt.title(f"Image {idx}\nMax: {cropped_data[idx].max():.2f}")
plt.tight_layout()
plt.show()
# # 对每一张图像进行裁剪
# cropped_data = []
#
# for img in data:
#     cropped_img = crop_around_max_no_pad(img, crop_size=64)
#     cropped_data.append(cropped_img.numpy())  # 转回numpy格式
#
# # 将列表转为numpy数组（N × 64 × 64）
# cropped_data = np.stack(cropped_data, axis=0)
#
# # 保存为新的npz文件
# output_path = 'cropped_inverse_scatter.npz'
# np.savez(output_path, cropped_data)
#
# print(f"裁剪完成！共处理 {len(data)} 张图像，保存至 {output_path}")
# print(f"输出数据形状: {cropped_data.shape} (N × H × W)")
# data = transforms.Resize((64,64), transforms.InterpolationMode.BICUBIC)(data).numpy()
# data = (data-data.min())/(data.max()-data.min())
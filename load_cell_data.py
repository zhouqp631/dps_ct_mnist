import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

# 设置matplotlib后端为Agg，避免在某些IDE下的问题
plt.switch_backend('Agg')


def crop_around_max_no_pad(image, crop_size=64, min_max_value=200):
    """
    以最大值位置为中心附近裁剪固定大小的区域，不填充，自动调整位置防止越界。
    如果图像的最大值小于指定的阈值，则跳过该图像。

    参数:
        image: 输入的2D张量（H × W）
        crop_size: 裁剪的目标尺寸（默认64×64）
        min_max_value: 最大值阈值，低于该值认为该图像没有足够的特征
    返回:
        cropped_image: 裁剪后的图像（crop_size × crop_size），若不满足阈值则返回 None
    """
    assert len(image.shape) == 2, "输入应为2D张量（H × W）"
    image = image.to(torch.float32)
    # 检查图像的最大值，如果小于阈值则跳过
    if torch.max(image).item() < min_max_value:
        return None

    h, w = image.shape
    half_size = crop_size // 2

    # 找到最大值位置（取第一个）
    max_pos = (image == torch.max(image)).nonzero()[0]
    max_y, max_x = max_pos[0].item(), max_pos[1].item()

    # 初始裁剪区域（可能越界）
    y_start = max_y - half_size
    y_end = max_y + half_size
    x_start = max_x - half_size
    x_end = max_x + half_size

    # 调整越界的裁剪区域
    if y_start < 0:
        y_start = 0
        y_end = crop_size
    if y_end > h:
        y_end = h
        y_start = h - crop_size
    if x_start < 0:
        x_start = 0
        x_end = crop_size
    if x_end > w:
        x_end = w
        x_start = w - crop_size

    # 执行裁剪
    cropped = image[y_start:y_end, x_start:x_end]

    # 检查是否成功裁剪到目标尺寸
    assert cropped.shape == (crop_size, crop_size), \
        f"裁剪失败，得到 {cropped.shape}，但期望 {crop_size}x{crop_size}"

    return cropped


# 加载原始数据
data = np.load('inverse_scatter.npz')['arr_0']
print(f"数据形状: {data.shape} (N × H × W)")

# 对每一张图像进行裁剪并筛选
cropped_data = []

for img in data:
    img_tensor = torch.tensor(img)  # 将numpy数组转为torch张量
    cropped_img = crop_around_max_no_pad(img_tensor, crop_size=64, min_max_value=200)

    if cropped_img is not None:  # 仅保留有效图像
        cropped_data.append(cropped_img.numpy())  # 转回numpy格式

# 将列表转为numpy数组（N × 64 × 64）
cropped_data = np.stack(cropped_data, axis=0)

# 保存为新的npz文件
output_path = 'cropped_filtered_inverse_scatter.npz'
np.savez(output_path, cropped_data)

print(f"裁剪完成！共处理 {len(cropped_data)} 张有效图像，保存至 {output_path}")
print(f"输出数据形状: {cropped_data.shape} (N × H × W)")

# 限制只显示前100张图像
n = min(1000, len(cropped_data))  # 确保最多显示100张
cols = 5  # 每行显示5张图像
rows = n // cols + (1 if n % cols != 0 else 0)  # 计算行数，确保能显示所有图像

# 创建画布
plt.figure(figsize=(15, 3 * rows))

for i in range(n):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(cropped_data[i], cmap='gray')
    plt.colorbar()
    plt.title(f"Image {i}\nMax: {cropped_data[i].max():.2f}")

plt.tight_layout()
plt.savefig('cropped_images_preview.png')  # 保存为PNG文件
# plt.show()  # 如果你希望在IDE中显示图像，可以取消注释这一行

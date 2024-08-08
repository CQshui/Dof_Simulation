# -*- coding: utf-8 -*-
"""
@ Time:     2024/8/8 14:54 2024
@ Author:   CQshui
$ File:     test.py
$ Software: Pycharm
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# 参数
wavelength = 0.633  # 激光波长，单位为微米
k = 2 * np.pi / wavelength  # 波数
num_points = 1024  # 全息图平面的点数
size = 100  # 全息图平面的物理尺寸，单位为微米
z_distance = 1000  # 从小球到全息图平面的距离，单位为微米

# 创建坐标网格
x = np.linspace(-size / 2, size / 2, num_points)
y = np.linspace(-size / 2, size / 2, num_points)
X, Y = np.meshgrid(x, y)

# 生成示例数据（小球）
np.random.seed(10)
num_spheres = 50
sphere_positions = np.random.uniform(-size / 2, size / 2, (num_spheres, 3))
sphere_radii = np.random.uniform(5, 15, num_spheres)  # 小球半径，单位为微米

# 模拟参考光束（平面波）
reference_beam = np.exp(1j * k * z_distance)

# 模拟物光束（散射波）
object_beam = np.zeros((num_points, num_points), dtype=np.complex128)

for i in range(num_spheres):
    x0, y0, z0 = sphere_positions[i]
    radius = sphere_radii[i]

    # 计算从小球到全息图平面各点的距离
    distance = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2 + z_distance ** 2)

    # 添加球面波贡献
    object_beam += np.exp(1j * k * distance) / distance * np.exp(-(X - x0) ** 2 / (2 * radius ** 2)) * np.exp(
        -(Y - y0) ** 2 / (2 * radius ** 2))

# 组合参考光束和物光束形成全息图
hologram = np.abs(reference_beam + object_beam) ** 2

# 绘制全息图
plt.imshow(hologram, cmap='gray', extent=(-size / 2, size / 2, -size / 2, size / 2))
plt.colorbar()
plt.title('离轴数字全息图')
plt.xlabel('X (微米)')
plt.ylabel('Y (微米)')
plt.show()

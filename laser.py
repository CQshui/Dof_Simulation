# -*- coding: utf-8 -*-
"""
@ Time:     2024/8/8 14:38 2024
@ Author:   CQshui
$ File:     laser.py
$ Software: Pycharm
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from show_3D import data_generator

import matplotlib
matplotlib.use('TkAgg')


def laser_simulation():
    # 模拟参考光束（平面波）
    reference_beam = np.exp(1j * k * z_distance)
    # 模拟物光束（散射波）
    object_beam = np.zeros((num_pix, num_pix), dtype=np.complex128)

    for i in range(num_objects):
        x0 = data[0][i]
        y0 = data[1][i]
        z0 = data[2][i] * pix
        radius = data[3][i] * pix

        # 计算从小球到全息图平面各点的距离
        distance = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2 + (z_distance - z0) ** 2)

        # 添加球面波贡献
        object_beam += np.exp(1j * k * distance) / distance * np.exp(-(X - x0) ** 2 / (2 * radius ** 2)) * np.exp(
            -(Y - y0) ** 2 / (2 * radius ** 2))

    # 组合参考光束和物光束形成全息图
    hologram = np.abs(reference_beam + object_beam) ** 2

    # 绘制全息图
    plt.imshow(hologram, cmap='gray', extent=(0, size, 0, size))
    plt.colorbar()
    plt.title('Hologram')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

    plt.imsave('holo.png', hologram)


    return None


if __name__ == '__main__':
    # 参数
    wavelength = 532e-9  # 激光波长
    k = 2 * np.pi / wavelength  # 波数

    num_objects = 2  # 小球个数
    z_distance = 1e-3  # 从物平面中心到全息图平面的距离

    pix = 0.098e-6  # 像素尺寸
    num_pix = 1024  # 全息图平面的点数
    size = pix * num_pix  # 全息图物理尺寸，此处视全息图为正方形

    # 创建坐标网格
    x = np.linspace(0, size, num_pix)
    y = np.linspace(0, size, num_pix)
    X, Y = np.meshgrid(x, y)

    # 引入小球数据
    data = data_generator(X_size=size, Y_size=size*4, num=num_objects)  # 创建小球

    laser_simulation()

# -*- coding: utf-8 -*-
"""
@ Time:     2024/8/8 14:07 2024
@ Author:   CQshui
$ File:     generator.py
$ Software: Pycharm
"""

import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def data_generator(num=50, X_size=100, Y_size=100, depth=10):
    # 生成示例数据
    np.random.seed(101)

    # 生成数组
    X = np.random.uniform(0, X_size, num)
    Y = np.random.uniform(0, Y_size, num)
    Z = np.random.uniform(0, depth, num)
    sizes = np.random.uniform(10, 50, size=num)  # 生成不同的大小

    # 创建三维散点图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, Z, s=sizes, c='r', marker='o')  # s参数设置大小
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.title('3D Scatter Plot with Varying Sizes')
    plt.show()

    return X, Y, Z, sizes


if __name__ == '__main__':
    data_generator()

"""
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0)  # n_clusters 簇的个数
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def showtable(data, plt):
    """
    绘制散点图

    :param data: 数据集
    :param plt: 绘图模块
    """
    # 分割数据集
    x = data.T[0]
    y = data.T[1]
    plt.scatter(x, y)

def kmeans(data, n, m, k, plt):
    """
    k-means算法实现

    :param data: 数据集
    :param n: 数据个数
    :param m: 数据维度
    :param k: 簇的个数
    :param plt: 绘图模块
    """
    rarray = np.random.random(size = k)  # 获取 k 个随机数
    rarray = np.floor(rarray*n)          # 乘以数据集大小——>数据集中随机的 n 个点
    rarray = rarray.astype(int)          # 转为 int
    print('数据集中随机索引', rarray)
    center = data[rarray]                # 随机取数据集中的 k 个点作为初始中心点

    cls = np.zeros([n], np.int)
    print('初始center=\n', center)
    run = True
    time = 0    # 迭代次数
    while run:
        time = time + 1
        for i in range(n):
            tmp = data[i] - center    # 求差
            tmp = np.square(tmp)      # 求平方
            tmp = np.sum(tmp, axis=1) # axis=1表示按行求和
            cls[i] = np.argmin(tmp)   # 取最小（最近）的给该点“染色”（标记每个样本所属的类(k[i])）
        run = False   # 如果没有修改各分类中心点，就结束循环

        # 计算更新每个类的中心点
        for i in range(k):
            club = data[cls==i]                # 找到属于该类的所有样本
            newcenter = np.mean(club, axis=0)  # axis=0表示按列求平均值，计算出新的中心点

            # 如果新旧center的差距很小，看做他们相等，否则更新之。run置true，再来一次循环
            ss = np.abs(center[i]-newcenter)
            if np.sum(ss, axis=0) > 0.1:
                center[i] = newcenter
                run = True
        print('new center=\n', center)
        if time >= 1000:    # 最大迭代次数设置为1000
            run = False
    print('程序结束，迭代次数：', time)

    # 按类打印图表，因为每打印一次，颜色都不一样，所以可区分出来
    for i in range(k):
        club = data[cls == i]
        showtable(club, plt)
    showtable(center, plt)  # 打印最后的中心点

if __name__ == "__main__":
    df = pd.read_csv(".\\K-means.data")
    kmeans(df.values, 221, 2, 3, plt)
    plt.show()
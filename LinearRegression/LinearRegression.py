"""
from sklearn.linear_model import LinearRegression
model = LinearRegression()
"""

import numpy as np
import matplotlib.pyplot as plt

# 梯度下降法实现
def liner_loss(w,b,data):
    """
    计算loss

    :param w: 权重
    :param b: 偏置
    :param data: 数据集
    :return: 均方误差MES
    """
    x = data[:,0]  # 代表的是第一列数据
    y = data[:,1]  # 代表的是第二列数据
    loss = np.sum((y - w * x - b) ** 2) / data.shape[0]
    
    return loss
 
def liner_gradient(w,b,data,lr):
    """
    计算梯度并更新参数

    :param w: 权重
    :param b: 偏置
    :param data: 数据集
    :param lr: 学习率 learning rate
    :return: 一个迭代后的 w 和 b 
    """
    # 数据集行数
    N = float(len(data))
    # 提取数据
    x = data[:,0]
    y = data[:,1]
    # 求梯度
    dw = np.sum(-(2 / N) * x * (y - w * x -b))
    db = np.sum(-(2 / N) * (y - w * x -b))
    # 更新参数
    w = w - (lr * dw)
    b = b - (lr * db)
 
    return w,b
 
 
def optimer(data,w,b,lr,epcoh):
    """
    每次迭代做梯度下降

    :param data: 数据集
    :param w: 权重
    :param b: 偏置
    :param lr: 学习率
    :param epcoh: 训练的次数
    :return: 更新后的 w 和 b 
    """
 
    for i in range(epcoh):
        # 通过每次循环不断更新w,b的值
        w,b = liner_gradient(w,b,data,lr)
        # 每训练100次更新下loss值
        if i % 100 == 0 :
            print('epoch {0}:loss={1}'.format(i,liner_loss(w,b,data)))
 
    return w,b
 

def plot_data(data,w,b):
    """
    绘图

    :param data: 数据集
    :param w: 权重
    :param b: 偏置
    """
    x = data[:,0]
    y = data[:,1]
    y_predict = w * x + b
    plt.plot(x, y, 'o')
    plt.plot(x, y_predict, 'k-')
    plt.show()

def liner_regression():
    """
    构建模型
    """
    # 加载数据
    data = np.loadtxt('.\\LinearRegression.csv',delimiter=',')
    # 显示原始数据的分布
    x = data[:, 0]
    y = data[:, 1]
    plt.plot(x, y, 'o')
    plt.show()
 
    # 初始化参数
    lr = 0.01     # 学习率
    epoch = 1000  # 训练次数
    w = 0.0       # 权重
    b = 0.0       # 偏置
    # 输出各个参数初始值
    print('initial variables:\n initial_b = {0}\n intial_w = {1}\n loss of begin = {2} \n'\
        .format(b,w,liner_loss(w,b,data)))

    # 更新w和b
    w,b = optimer(data,w,b,lr,epoch)
 
    # 输出各个参数的最终值
    print('final formula parmaters:\n b = {0}\n w = {1}\n loss of end = {2} \n'.format(b, w, liner_loss(w, b, data)))
    # 显示
    plot_data(data,w,b)



# 最小二乘法实现
class LinerRegressionModel(object):
    def __init__(self, data):
        """
        初始化参数

        """
        self.data = data
        self.x = data[:, 0]
        self.y = data[:, 1]

    def log(self, a, b):
        """
        打印日志

        :param data: 数据集
        :param w: 权重
        :param b: 偏置
        """
        print("计算出的线性回归函数为:\ny = {:.5f}x + {:.5f}".format(a, b))

    def plt(self, a, b):
        """
        画图


        """
        plt.plot(self.x, self.y, 'o', label='data', markersize=10)
        plt.plot(self.x, a * self.x + b, 'r', label='line')
        plt.legend()
        plt.show()

    def least_square_method(self):
        """
        最小二乘法的实现
        """
        def calc_ab(x, y):
            sum_x, sum_y, sum_xy, sum_xx = 0, 0, 0, 0
            n = len(x)
            for i in range(0, n):
                sum_x += x[i]
                sum_y += y[i]
                sum_xy += x[i] * y[i]
                sum_xx += x[i]**2
            a = (sum_xy - (1/n) * (sum_x * sum_y)) / (sum_xx - (1/n) * sum_x**2)
            b = sum_y/n - a * sum_x/n
            return a, b
        a, b = calc_ab(self.x, self.y)
        self.log(a, b)
        self.plt(a, b)



if __name__ == "__main__":
    # 梯度下降法
    liner_regression()

    # 最小二乘法
    # data = np.array([[1, 2.5], [2, 3.3], [2.5, 3.8],[3, 4.5], [4, 5.7], [5, 6]])
    # model = LinerRegressionModel(data)
    # model.least_square_method()

import numpy as np

def loaddataset():
    """
    加载数据

    :param filename: 文件名
    :return dataset: 数据集
    :return labelset: 标签集
    """
    raw_data = np.genfromtxt('./abalone_train.csv', dtype='str', comments='%', delimiter=',')
    raw_data[raw_data == 'M'] = '0'
    raw_data[raw_data == 'F'] = '1'
    raw_data[raw_data == 'I'] = '2'
    dataset = raw_data[:, 1:].astype('float')
    labelset = raw_data[:, 0].astype('int')

    return dataset, labelset

def parameter_initialization(x, y, z):
    """
    参数初始化

    :param x: 输入层神经元个数
    :param y: 隐层神经元个数
    :param z: 输出层神经元个数
    :return weight1: 输入层与隐层的连接权重
    :return weight2: 隐层与输出层的连接权重
    :return value1: 隐藏层阈值
    :return value2: 输出层阈值
    """
    weight1 = np.random.randint(-5, 5, (x, y)).astype(np.float64)
    weight2 = np.random.randint(-5, 5, (y, z)).astype(np.float64)
    value1 = np.random.randint(-5, 5, (1, y)).astype(np.float64)
    value2 = np.random.randint(-5, 5, (1, z)).astype(np.float64)
    return weight1, weight2, value1, value2
 
def sigmoid(z):
    """
    sigmod 函数
    """
    return 1 / (1 + np.exp(-z))
 

def trainning(dataset, labelset, weight1, weight2, value1, value2):
    '''
    :param dataset: 数据集，不含标签
    :param labelset: 标签集
    :param weight1: 输入层与隐层的连接权重
    :param weight2: 隐层与输出层的连接权重
    :param value1: 隐层阈值
    :param value2: 输出层阈值

    :return weight1: 更新后的，输入层与隐层的连接权重
    :return weight2: 更新后的，隐层与输出层的连接权重
    :return value1: 更新后的，隐层阈值
    :return value2: 更新后的，输出层阈值
    '''
    x = 0.01    # x为步长
    for i in range(len(dataset)):
        inputset = np.mat(dataset[i]).astype(np.float64)       # 输入数据
        outputset = np.mat(labelset[i]).astype(np.float64)     # 数据标签
        input1 = np.dot(inputset, weight1).astype(np.float64)  # 隐层输入
        output2 = sigmoid(input1 - value1).astype(np.float64)  # 隐层输出
        input2 = np.dot(output2, weight2).astype(np.float64)   # 输出层输入
        output3 = sigmoid(input2 - value2).astype(np.float64)  # 输出层输出
 
        #更新公式由矩阵运算表示
        a = np.multiply(output3, 1 - output3)
        g = np.multiply(a, outputset - output3)
        b = np.dot(g, np.transpose(weight2))
        c = np.multiply(output2, 1 - output2)
        e = np.multiply(b, c)
 
        value1_change = -x * e
        value2_change = -x * g
        weight1_change = x * np.dot(np.transpose(inputset), e)
        weight2_change = x * np.dot(np.transpose(output2), g)
 
        #更新参数
        value1 += value1_change
        value2 += value2_change
        weight1 += weight1_change
        weight2 += weight2_change
    return weight1, weight2, value1, value2
 
def testing(dataset, labelset, weight1, weight2, value1, value2):
    """
    测试

    :param dataset: 数据集，不含标签
    :param labelset: 标签集
    :param weight1: 输入层与隐层的连接权重
    :param weight2: 隐层与输出层的连接权重
    :param value1: 隐层阈值
    :param value2: 输出层阈值
    :return : 正确率
    """
    rightcount = 0   # 记录预测正确的个数
    for i in range(len(dataset)):
        # 计算每一个样例通过该神经网路后的预测值
        inputset = np.mat(dataset[i]).astype(np.float64)
        outputset = np.mat(labelset[i]).astype(np.float64)
        output2 = sigmoid(np.dot(inputset, weight1) - value1)
        output3 = sigmoid(np.dot(output2, weight2) - value2)
 
        #确定其预测标签
        if output3 > 0.5:
            flag = 1
        else:
            flag = 0
        if labelset[i] == flag:
            rightcount += 1
        print("预测为%d   实际为%d"%(flag, labelset[i]))
    # 返回正确率
    return rightcount / len(dataset)
 
if __name__ == '__main__':
    dataset, labelset = loaddataset()
    weight1, weight2, value1, value2 = parameter_initialization(len(dataset[0]), len(dataset[0]), 1)
    for i in range(500):  # 迭代次数
        weight1, weight2, value1, value2 = trainning(dataset, labelset, weight1, weight2, value1, value2)
    rate = testing(dataset, labelset, weight1, weight2, value1, value2)
    print("correct rate is: %f"%(rate))
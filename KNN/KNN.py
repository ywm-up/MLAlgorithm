"""
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
"""
from numpy import *
import operator

#定义KNN算法分类器函数
#函数参数包括：(测试数据，训练数据，分类, k值)
def classify(inX, dataSet, labels, k):
    """
    KNN分类算法

    :param inx: 测试数据
    :param dataSet: 数据集
    :param labels: 标签集
    :param k: K值
    :return sortedClassCount[0][0]: 分类结果
    """
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances ** 0.5 # 计算欧式距离
    sortedDistIndicies = distances.argsort() # 排序并返回index

    # 选择距离最近的k个值
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        #D.get(k[,d]) -> D[k] if k in D, else d. d defaults to None.
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
    #排序
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True) # 降序
    return sortedClassCount[0][0]  # 将出现次数最多的标签输出


#定义一个生成“训练样本集”的函数，包含特征和分类信息
def createDataSet():
    """
    生成样本集

    :return group: 数据集
    :return labels: 标签集
    """
    group = array([[1,1.1], [1,1], [0,0], [0,0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

if __name__ == "__main__":
    # 生成训练样本
    group, labels = createDataSet()

    # 对测试数据[0， 0]进行KNN分类
    print('classfication result is: {0}'.format(classify([0,0], group, labels, 3)))
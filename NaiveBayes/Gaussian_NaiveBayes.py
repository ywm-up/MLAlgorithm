"""
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
"""
import pandas as pd
import numpy as np
import os

data=pd.read_csv(r".\\pima-indians-diabetes.data.csv",header=None,\
                 names=["怀孕次数","口服葡萄糖耐量试验中血浆葡萄糖浓度",\
                        "舒张压（mm Hg）","三头肌组织褶厚度（mm）",\
                        "2小时血清胰岛素（μU/ ml）","体重指数（kg/（身高(m)）^ 2）",\
                        "糖尿病系统功能","年龄（岁）","是否患有糖尿病"])

data.iloc[:,0:8] = data.iloc[:,0:8].applymap(lambda x:np.NaN if x == 0 else x) #把属性值为0的地方转换成NaN

data = data.dropna(how="any",axis=0) #去除有缺失值的行

#随机选取80%的样本作为训练样本
data_train = data.sample(frac = 0.8, random_state = 4, axis=0)
#剩下的作为测试样本
test_idx = [i for i in data.index.values if i not in data_train.index.values]
data_test = data.loc[test_idx,:]

#提取训练集和测试集的特征和目标
X_train = data_train.iloc[:,:-1]
y_train = data_train.iloc[:,-1]

X_test = data_test.iloc[:,:-1]
y_test = data_test.iloc[:,-1]


class Gaussian_NB:
    def __init__(self):
        """
        初始化变量
        """
        self.num_of_samples = None
        self.num_of_class = None
        self.class_name = []
        self.prior_prob = []
        self.X_mean = []
        self.X_var = []

    def SepByClass(self, X, y):
        """
        按类别分割数据

        :param X: 数据特征
        :param y: 数据标签
        :return data_byclass: 按类别分类的字典
        """

        self.num_of_samples = len(y) # 总样本数
        y = y.reshape(X.shape[0],1)
        data = np.hstack((X,y))        # 把特征和目标合并成完整数据
        data_byclass = {}              # 初始化分类数据，为一个空字典

        # 提取各类别数据，字典的键为类别名，值为对应的分类数据
        for i in range(len(data[:,-1])):
            if i in data[:,-1]:
                data_byclass[i] = data[data[:,-1]==i]
        
        self.class_name = list(data_byclass.keys())  # 类别名
        self.num_of_class = len(data_byclass.keys()) # 类别总数
        
        return data_byclass
    
    def CalPriorProb(self, y_byclass):
        """
        计算 y 的先验概率（使用拉普拉斯平滑）

        :param y_byclass: 当前类别下的目标
        :return : 该目标的先验概率
        """
        # 计算公式：（当前类别下的样本数+1）/（总样本数+类别总数）
        return (len(y_byclass)+1)/(self.num_of_samples+self.num_of_class)
    
    def CalXMean(self, X_byclass):
        """
        计算各类别特征各维度的平均值

        :param X_byclass: 当前类别下的特征
        :return X_mean: 该特征各个维度的平均值
        """
        X_mean = []
        for i in range(X_byclass.shape[1]):
            X_mean.append(np.mean(X_byclass[:,i]))
        return X_mean

    def CalXVar(self, X_byclass):
        """
        计算各类别特征各维度的方差

        :param X_byclass: 当前类别下的特征
        :return X_var: 该特征各个维度的方差
        """
        X_var = []
        for i in range(X_byclass.shape[1]):
            X_var.append(np.var(X_byclass[:,i]))
        return X_var
    
    def CalGaussianProb(self, X_new, mean, var):
        """
        计算训练集特征（符合正态分布）在各类别下的条件概率

        :param X_new: 新样本的特征
        :param mean: 训练集特征的平均值
        :param var: 训练集特征的方差
        :return gaussian_prob: 新样本的特征在相应训练集中的分布概率
        """
        # 计算公式：(np.exp(-(X_new-mean)**2/(2*var)))*(1/np.sqrt(2*np.pi*var))
        gaussian_prob = []
        for a,b,c in zip(X_new, mean, var):
            formula1 = np.exp(-(a-b)**2/(2*c))
            formula2 = 1/np.sqrt(2*np.pi*c)
            gaussian_prob.append(formula2*formula1)
        return gaussian_prob

    def fit(self, X, y):
        """
        训练数据

        :param X: 训练集特征
        :param y: 训练集标签
        :return prior_prob: 标签的先验概率
        :return X_mean: 平均值
        :return X_var: 方差
        """

        # 将输入的X,y转换为numpy数组
        X, y = np.asarray(X, np.float32), np.asarray(y, np.float32)      
        data_byclass = Gaussian_NB.SepByClass(X,y) # 将数据分类

        # 计算各类别数据的目标先验概率，特征平均值和方差
        for data in data_byclass.values():
            X_byclass = data[:,:-1]
            y_byclass = data[:,-1]
            self.prior_prob.append(Gaussian_NB.CalPriorProb(y_byclass))
            self.X_mean.append(Gaussian_NB.CalXMean(X_byclass))
            self.X_var.append(Gaussian_NB.CalXVar(X_byclass))
        
        return self.prior_prob, self.X_mean, self.X_var
        
    def predict(self,X_new):
        """
        预测数据

        :param X_new: 新样本的特征
        :return class_name[idx]: 新样本最有可能的标签
        """
        # 将输入的x_new转换为numpy数组
        X_new = np.asarray(X_new, np.float32)
        
        posteriori_prob = [] # 初始化极大后验概率
        
        for i,j,o in zip(self.prior_prob, self.X_mean, self.X_var):
            gaussian = Gaussian_NB.CalGaussianProb(X_new,j,o)
            posteriori_prob.append(np.log(i)+sum(np.log(gaussian)))
            idx = np.argmax(posteriori_prob)
        
        return self.class_name[idx]
           
if __name__=="__main__":
    Gaussian_NB = Gaussian_NB()      # 实例化Gaussian_NB
    Gaussian_NB.fit(X_train,y_train) # 使用Gaussian_NB模型训练数据
    acc=0
    TP=0
    FP=0
    FN=0
    for i in range(len(X_test)):
        predict = Gaussian_NB.predict(X_test.iloc[i,:])
        target = np.array(y_test)[i]
        if predict == 1 and target == 1:
            TP += 1
        if predict == 0 and target == 1:
            FP += 1
        if predict == target:
            acc += 1
        if predict == 1 and target == 0:
            FN += 1
    print("准确率:",acc/len(X_test))
    print("查准率:",TP/(TP+FP))
    print("查全率:",TP/(TP+FN))
    print("F1:",2*TP/(2*TP+FP+FN))
    os.system("pause")
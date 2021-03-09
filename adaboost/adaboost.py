import numpy as np
import cv2
from sklearn.tree import DecisionTreeClassifier
import os
import random
import argparse

def load_data(root_dir, is_training=True):
    """
    读取图片集

    :param root_dir: 图片集路径
    :param is_training: 是否是训练集，默认为是
    :return imgs[]: 返回图像的灰度图
    :return lbs[]: 返回图片标签列表
    """
    def load_img(img_pts):
        """

        :param img_pts: 图片输入
        :return imgs:
        """
        imgs = np.zeros([len(img_pts), 14 * 14])
        for i, each in enumerate(img_pts):
            """
            cv2.imread()返回np.array类型，代表高度、宽度、通道数的元组
            通道数顺序为BGR
            """
            imgs[i] = cv2.imread(each)[:, :, 0].reshape(1, -1)
        return imgs

    def load_lbl(img_pts):
        """
        得到图像的标签，数据集中图片名自带了标签，从图片名中可以取出标签

        :param img_pts: 图片输入
        
        :return lbs: 标签列表
        """
        lbs = np.zeros(len(img_pts))
        for i, each in enumerate(img_pts):
            """
            split("\\")[-1] 分割出路径，取出了文件名+后缀
            split("_")[-1]  取出了标签和后缀
            split(".")[0]   取出了标签
            """
            lb = int(each.split("\\")[-1].split("_")[-1].split(".")[0])
            # 二分类问题，0和非0
            if lb == 0:
                lbs[i] = -1
            else:
                lbs[i] = 1
        return lbs

    img_pts = [os.path.join(root_dir, each) for each in os.listdir(root_dir)]
    imgs = load_img(img_pts)
    lbls = load_lbl(img_pts)
    random_list = list(range(imgs.shape[0]))
    random.shuffle(random_list)
    return imgs[random_list], lbls[random_list]


def adaboost(X_train, Y_train, X_test, M=10, clf=DecisionTreeClassifier(max_depth=1)):
    """
    adaboost 训练模型

    :param X_train: 训练集图片
    :param Y_train: 训练集标签
    :param X_test: 测试集图片
    :param M: 迭代次数
    :param clf: 分类器，预定义为一层决策树

    :return pred_train: 训练准确率
    :return pred_test: 测试准确率
    """
    n_train, n_test = X_train.shape[0], X_test.shape[0]
    # 初始化权值
    w = np.ones(n_train) / n_train
    pred_train, pred_test = [np.zeros(n_train), np.zeros(n_test)]
    for i in range(M):
        print(i)
        # 通过权重拟合分类器
        clf.fit(X_train, Y_train, sample_weight=w)
        pred_train_i = clf.predict(X_train)
        pred_test_i = clf.predict(X_test)
        print(np.sum(pred_train_i == Y_train) / pred_train_i.shape[0])

        # 评价指标函数
        miss = [int(x) for x in (pred_train_i != Y_train)]
        # 和 1/-1 等效以更新权重
        miss2 = [x if x == 1 else -1 for x in miss]
        # 该弱分类器在分布Dt上的Error（误差），即e = P(Ht(xi) != yi)
        err_m = np.dot(w, miss) / sum(w) + 0.00001
        # 该弱分类器在最终分类器中所占的权重Alpha, alpha = 1/2 * ln[(1-et)/et]
        alpha_m = 0.5 * np.log((1 - err_m) / float(err_m))

        # 新权重,更新训练样本的权值分布Dt+1
        w = np.multiply(w, np.exp([float(x) * alpha_m for x in miss2]))
        w = w / sum(w)
        # 添加到预测中
        pred_train_i = [1 if x == 1 else -1 for x in pred_train_i]
        pred_test_i = [1 if x == 1 else -1 for x in pred_test_i]
        pred_train = pred_train + np.multiply(alpha_m, pred_train_i)
        pred_test = pred_test + np.multiply(alpha_m, pred_test_i)
    pred_train, pred_test = np.sign(pred_train), np.sign(pred_test)

    return pred_train, pred_test

if __name__ == "__main__":
    train_dir = './train'
    test_dir = './test'

    X_train, Y_train = load_data(train_dir, is_training=True)
    X_test,  Y_test  = load_data(test_dir, is_training=False)
    pred_train, pred_test = adaboost(X_train, Y_train, X_test, 20)

    print(np.sum(pred_test == Y_test)/ pred_test.shape[0])
    print(np.sum(Y_train == pred_train)/ pred_train.shape[0])
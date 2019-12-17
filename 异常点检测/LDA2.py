# -*- coding: utf-8 -*-
"""
Created on Wed May 15 14:48:32 2019
Python 3.6.8
@author: yh
"""
from numpy import linalg as LA
from scipy import io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import tensorflow as tf

"""
LDA线性判别分析
数据降维+KNN分类
"""

np.random.seed(0)
"""
导入数据
这个数据集1400X1024
20X72
20中类别,每类72个32x32的灰度图像
"""
data = io.loadmat('COIL20.mat')
x = data['fea']
y = data['gnd']


def Std(Sw):
    u = np.mean(Sw)
    s = np.std(Sw)
    Sw = (Sw - u) / s
    return Sw


# X是数据集,k是降到k维,n是数据集中类的个数,m是每一个样本的维度
def Fisher(X, k, n, m):
    # 求每一类的均值
    mu = np.zeros((n, 1, m))
    for i in range(n):
        for j in range(len(X[i])):
            mu[i] = mu[i] + X[i][j]
        mu[i] = mu[i] / len(X[i])
    # 总体均值
    u = np.mean(mu, axis=0)
    Sw = np.zeros((m, m))
    Sb = np.zeros((m, m))
    for i in range(n):
        for j in range(len(X[i])):
            # 注意np.dot才是矩阵乘法,‘*’对于array来说是点乘
            Sw = Sw + np.dot((X[i][j] - mu[i]).T, (X[i][j] - mu[i]))
    for i in range(n):
        Sb = Sb + np.dot((mu[i] - u).T, (mu[i] - u))
    # 单位矩阵
    I = np.identity(m)
    # 归一化
    Sw = Std(Sw);
    Sw = Sw + 0.000000001 * I
    val, vec = np.linalg.eig(np.dot(np.mat(Sw).I, Sb))
    # 求最大K个特征向量编号
    index = np.argsort(-val)
    # 最大K个特征向量所组成的矩阵
    w = vec[index[0:k]]
    # 取w矩阵实部返回
    return w.real


def Class_KNN(X, Y):
    # KNN分类
    # 自动分出训练集,测试集
    x_train, x_test, y_train, y_test = train_test_split(X, Y)
    knn = KNeighborsClassifier()
    knn.fit(x_train, y_train.ravel())
    accuracy = knn.score(x_test, y_test, sample_weight=None)
    return accuracy


def Showimage(X, x, n, m):
    ###这里可以修改最大降维维度
    k = 100
    K = np.arange(1, k)
    Ac = np.zeros((1, k - 1))
    print(K.shape, Ac.shape)
    for i in range(len(K)):
        w = Fisher(X, K[i], n, m)
        z = np.dot(x, w.T)
        Ac[0][i] = Class_KNN(z, y)
    # 画出k-ac图片
    plt.plot(K, Ac.ravel(), '-r')
    plt.xlabel('K-Dimesions')
    plt.ylabel('Accuracy')


if __name__ == "__main__":
    n, m = x.shape
    print(n, m)
    X = np.zeros((n // 72, 72, m))
    for i in range(n // 72):
        xi = x[i * 72:i * 72 + 72, :]
        X[i] = xi
    Showimage(X, x, n // 72, m)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_classification
from mpl_toolkits.mplot3d import Axes3D


def LDA(X, y):
    X1 = np.array([X[i] for i in range(len(X)) if y[i] == 0])
    X2 = np.array([X[i] for i in range(len(X)) if y[i] == 1])

    len1 = len(X1)
    len2 = len(X2)

    mju1 = np.mean(X1, axis=0)  # 求中心点
    mju2 = np.mean(X2, axis=0)

    cov1 = np.dot((X1 - mju1).T, (X1 - mju1))

    cov2 = np.dot((X2 - mju2).T, (X2 - mju2))
    Sw = cov1 + cov2

    w = np.dot(np.mat(Sw).I, (mju1 - mju2).reshape((len(mju1), 1)))# 计算w
    X1_new = func(X1, w)
    X2_new = func(X2, w)
    y1_new = [1 for i in range(len1)]
    y2_new = [2 for i in range(len2)]
    return X1_new, X2_new, y1_new, y2_new


def func(x, w):
    return np.dot((x), w)


if '__main__' == __name__:
    X, y = make_classification(n_samples=500, n_features=2, n_redundant=0, n_classes=2,
                               n_informative=1, n_clusters_per_class=1, class_sep=0.5, random_state=10)

    X1_new, X2_new, y1_new, y2_new = LDA(X, y)

    plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)
    plt.show()

    plt.plot(X1_new, y1_new, 'b*')
    plt.plot(X2_new, y2_new, 'ro')
    plt.show()


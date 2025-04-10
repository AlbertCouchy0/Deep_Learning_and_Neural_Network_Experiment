# -*- coding: utf-8 -*-

import scipy.misc, scipy.io, scipy.optimize
from sklearn import svm
from sklearn import model_selection
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 导入本地库
from Fbase import plot
from Fbase import visualize_boundary

# 高斯核函数
def gaussian_kernel(x1, x2, sigma):
    # your code here    2）
    # 把公式补充完整 return np.exp(-sum((x1 - x2) ** 2.0) / ?)
    return np.exp(-sum((x1 - x2) ** 2.0) / (2 * (sigma ** 2.0)))

def train_sigma(X,y,sigma):
    # 训练高斯核函数SVM
    rbf_svm = svm.SVC(C=1, kernel='rbf', gamma=1.0 / sigma)  # gamma is actually inverse of sigma
    rbf_svm.fit(X, y.ravel())

    # 查看训练集的错误率：
    predictions = []
    m = np.shape(X)[0]
    for i in range(0, m):
        prediction_result = rbf_svm.predict(X[i].reshape(-1, 2))
        predictions.append(prediction_result[0])
    # sadly if you don't reshape it, numpy doesn't know if it's row or column vector
    predictions = np.array(predictions).reshape(m, 1)
    error = (predictions != y.reshape(m, 1)).mean()
    print('When sigma={}, error of training set:'.format(sigma))
    print(error)

    # 绘制非线性SVM的决策边界
    plt.title('sigma={}时,非线性SVM的决策边界'.format(sigma))
    plot(np.c_[X, y])
    visualize_boundary(X, rbf_svm)
    plt.show(block=True)

# 非线性可分SVM
def part2():
    # --------------- 步骤1 ------------------
    # 计算高斯核函数
    x1 = np.array([1, 2, 1])
    x2 = np.array([0, 4, -1])
    sigma = 2
    print("样本x1和x2之间的相似度: %f" % gaussian_kernel(x1, x2, sigma))

    # --------------- 步骤2 ------------------
    # 加载数据集2
    mat = scipy.io.loadmat("dataset_2.mat")
    X, y = mat['X'], mat['y']
    # 绘制数据集2
    plt.title('数据集2分布')
    plot(np.c_[X, y])
    plt.show(block=True)

    # --------------- sigma = 0.01 ------------------
    train_sigma(X, y, 0.001)
    train_sigma(X, y, 0.003)
    train_sigma(X, y, 0.01)
    train_sigma(X, y, 0.03)
    train_sigma(X, y, 0.1)
    train_sigma(X, y, 1)


def main():
    np.set_printoptions(precision=6, linewidth=200)
    part2()


if __name__ == '__main__':
    main()

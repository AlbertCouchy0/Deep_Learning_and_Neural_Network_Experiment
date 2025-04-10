# -*- coding: utf-8 -*-

# 导入第三方库
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



# 线性可分SVM
def part1():
    # --------------- 步骤1 ------------------
    # 加载数据集1
    mat = scipy.io.loadmat("dataset_1.mat")
    X, y = mat['X'], mat['y']

    # 绘制数据集1
    plt.title('数据集1分布')
    plot(np.c_[X, y])
    plt.show(block=True)

    # --------------- （C = 1） ------------------
    # 训练线性SVM
    linear_svm: SVC = svm.SVC(C=1, kernel='linear')
    linear_svm.fit(X, y.ravel())

    # 绘制C=1的SVM决策边界
    plt.title('C=1的SVM决策边界')
    plot(np.c_[X, y])
    visualize_boundary(X, linear_svm)
    plt.show(block=True)

    # --------------- （C = 5） ------------------
    # 训练线性SVM
    linear_svm: SVC = svm.SVC(C=5, kernel='linear')
    linear_svm.fit(X, y.ravel())

    # 绘制C=1的SVM决策边界
    plt.title('C=5的SVM决策边界')
    plot(np.c_[X, y])
    visualize_boundary(X, linear_svm)
    plt.show(block=True)

    # --------------- （C = 25） ------------------
    # 训练线性SVM
    linear_svm: SVC = svm.SVC(C=25, kernel='linear')
    linear_svm.fit(X, y.ravel())

    # 绘制C=1的SVM决策边界
    plt.title('C=25的SVM决策边界')
    plot(np.c_[X, y])
    visualize_boundary(X, linear_svm)
    plt.show(block=True)

    # ---------------（C = 100）------------------
    # 训练线性SVM
    linear_svm: SVC = svm.SVC(C=100, kernel='linear')
    linear_svm.fit(X, y.ravel())

    # 绘制C=100的SVM决策边界
    # your code here  1）
    plt.title('C=100的SVM决策边界')
    plot(np.c_[X, y])
    visualize_boundary(X, linear_svm)
    plt.show(block=True)

    # ---------------（C = 400）------------------
    # 训练线性SVM
    linear_svm: SVC = svm.SVC(C=400, kernel='linear')
    linear_svm.fit(X, y.ravel())

    # 绘制C=100的SVM决策边界
    # your code here  1）
    plt.title('C=400的SVM决策边界')
    plot(np.c_[X, y])
    visualize_boundary(X, linear_svm)
    plt.show(block=True)

    # ---------------（C = 1000）------------------
    # 训练线性SVM
    linear_svm: SVC = svm.SVC(C=1000, kernel='linear')
    linear_svm.fit(X, y.ravel())

    # 绘制C=100的SVM决策边界
    # your code here  1）
    plt.title('C=1000的SVM决策边界')
    plot(np.c_[X, y])
    visualize_boundary(X, linear_svm)
    plt.show(block=True)


def main():
    np.set_printoptions(precision=6, linewidth=200)
    part1()


if __name__ == '__main__':
    main()

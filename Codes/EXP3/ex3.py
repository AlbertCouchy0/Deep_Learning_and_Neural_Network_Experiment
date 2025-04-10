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

# 高斯核函数
def gaussian_kernel(x1, x2, sigma):
    # your code here    2）
    # 把公式补充完整 return np.exp(-sum((x1 - x2) ** 2.0) / ?)
    return np.exp(-sum((x1 - x2) ** 2.0) / (2 * (sigma ** 2.0)))

# 导入本地库
from Fbase import plot
from Fbase import visualize_boundary
from Fpara import params_search



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

    # --------------- 步骤2 ------------------
    # 训练线性SVM（C = 1）
    linear_svm: SVC = svm.SVC(C=1, kernel='linear')
    linear_svm.fit(X, y.ravel())

    # 绘制C=1的SVM决策边界
    plt.title('C=1的SVM决策边界')
    plot(np.c_[X, y])
    visualize_boundary(X, linear_svm)
    plt.show(block=True)

    # --------------- 步骤3 ------------------
    # 训练线性SVM（C = 100）
    linear_svm: SVC = svm.SVC(C=100, kernel='linear')
    linear_svm.fit(X, y.ravel())

    # 绘制C=100的SVM决策边界
    # your code here  1）
    plt.title('C=100的SVM决策边界')
    plot(np.c_[X, y])
    visualize_boundary(X, linear_svm)
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
    # your code here  3)
    mat = scipy.io.loadmat("dataset_2.mat")
    X, y = mat['X'], mat['y']
    # 绘制数据集2
    # your code here  4)
    plt.title('数据集2分布')
    plot(np.c_[X, y])
    plt.show(block=True)
    # 训练高斯核函数SVM
    sigma = 0.01
    rbf_svm = svm.SVC(C=0.05, kernel='rbf', gamma=1.0 / sigma)  # gamma is actually inverse of sigma
    rbf_svm.fit(X, y.ravel())

    # 绘制非线性SVM的决策边界
    # 注意图标题的变化，visualize_boundary的第二个参数变化为rbf_svm
    # your code here  5)
    plt.title('非线性SVM的决策边界')
    plot(np.c_[X, y])
    visualize_boundary(X, rbf_svm)
    plt.show(block=True)


# 参数搜索
def part3():
    # --------------- 步骤1 ------------------
    # 加载数据集3和验证集
    mat = scipy.io.loadmat("dataset_3.mat")
    X, y = mat['X'], mat['y']
    X_val, y_val = mat['Xval'], mat['yval']

    # 绘制数据集3
    plt.title('数据集3的训练集分布')
    plot(np.c_[X, y])
    plt.show(block=True)

    # 绘制验证集
    plt.title('数据集3的验证集分布')
    plot(np.c_[X_val, y_val])
    plt.show(block=True)

    # 训练高斯核函数SVM并搜索使用最优模型参数
    rbf_svm = svm.SVC(kernel='rbf')
    # 通过rbf_svm.set_params可设定模型的C和gamma值
    y = y.ravel()
    best = params_search(X, y, X_val, y_val)
    print('Before improve:')
    print(best)
    rbf_svm.set_params(C=best['C'])
    rbf_svm.set_params(gamma=best['gamma'])
    rbf_svm.fit(X, y)

    # 绘制决策边界
    plt.title('参数搜索后的决策边界（优化前）')
    plot(np.c_[X, y])
    visualize_boundary(X, rbf_svm)
    plt.show(block=True)


def main():
    np.set_printoptions(precision=6, linewidth=200)
    # part1()
    # part2()
    part3()


if __name__ == '__main__':
    main()

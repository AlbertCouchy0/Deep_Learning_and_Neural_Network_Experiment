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
from FPpara import poly_params_search
from FPpara import poly_params_search_dual
from FPpara import poly_params_search_C
from FPpara import poly_params_search_sigma

def Poly_opt(X, y, X_val, y_val):
    # 搜索参数
    best1, best2 = poly_params_search_dual(X, y, X_val, y_val)
    print('Before Improve, ')
    print(best1)
    C_sec = best2['C']
    sigma_sec = best2['sigma']

    for i in range(1, 10):
        best1, C_sec = poly_params_search_C(X, y, X_val, y_val, best1, C_sec)
        print('\n Times {} of C optimization,'.format(i))
        print('val_error:{}; train_error:{}; C:{}; sigma:{};'.format(best1['val_error'], best1['train_error'], best1['C'], best1['sigma']))
        print('C_sec:{}'.format(C_sec))
        best1, sigma_sec = poly_params_search_sigma(X, y, X_val, y_val, best1, sigma_sec)
        print('\n Times {} of sigma optimization,'.format(i))
        print('val_error:{}; train_error:{}; C:{}; sigma:{};'.format(best1['val_error'], best1['train_error'], best1['C'],best1['sigma']))
        print('sigma_sec:{}'.format(sigma_sec))


def Poly_opt(X, y, X_val, y_val, degree=3):
    poly_svm = svm.SVC(kernel='poly')
    y = y.ravel()
    # 搜索参数
    best1, best2 = poly_params_search_dual(X, y, X_val, y_val, degree)
    print('Before Improve, ')
    print(best1)
    C_sec = best2['C']
    sigma_sec = best2['sigma']

    for i in range(1, 10):
        best1, C_sec = poly_params_search_C(X, y, X_val, y_val, best1, C_sec, degree)
        print('\n Times {} of C optimization,'.format(i))
        print('val_error:{}; train_error:{}; C:{}; sigma:{};'.format(best1['val_error'], best1['train_error'], best1['C'], best1['sigma']))
        print('C_sec:{}'.format(C_sec))
        best1, sigma_sec = poly_params_search_sigma(X, y, X_val, y_val, best1, sigma_sec, degree)
        print('\n Times {} of sigma optimization,'.format(i))
        print('val_error:{}; train_error:{}; C:{}; sigma:{};'.format(best1['val_error'], best1['train_error'], best1['C'],best1['sigma']))
        print('sigma_sec:{}'.format(sigma_sec))

    print('\n After Improve, ')
    print(best1)
    # 通过poly_svm.set_params可设定模型的C和gamma值
    poly_svm.set_params(degree=degree)
    poly_svm.set_params(C=best1['C'])
    poly_svm.set_params(gamma=best1['gamma'])
    poly_svm.fit(X, y)
    # 绘制决策边界
    plt.title('采取{}维多项式函数作为核函数的决策边界'.format(degree))
    plot(np.c_[X, y])
    visualize_boundary(X, poly_svm)
    plt.show(block=True)


def main():
    np.set_printoptions(precision=6, linewidth=200)
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

    # Poly_opt(X, y, X_val, y_val, 1)
    # Poly_opt(X, y, X_val, y_val, 2)
    # Poly_opt(X, y, X_val, y_val, 3)
    # Poly_opt(X, y, X_val, y_val, 4)


    degree = 4
    poly_svm = svm.SVC(kernel='poly',degree=degree)
    y = y.ravel()
    # 搜索参数
    best1, best2 = poly_params_search_dual(X, y, X_val, y_val, degree)
    print('Before Improve, ')
    print(best1)
    C_sec = best2['C']
    sigma_sec = best2['sigma']

    for i in range(1, 10):
        best1, C_sec = poly_params_search_C(X, y, X_val, y_val, best1, C_sec, degree)
        print('\n Times {} of C optimization,'.format(i))
        print(
            'val_error:{}; train_error:{}; C:{}; sigma:{};'.format(best1['val_error'], best1['train_error'], best1['C'],
                                                                   best1['sigma']))
        print('C_sec:{}'.format(C_sec))
        best1, sigma_sec = poly_params_search_sigma(X, y, X_val, y_val, best1, sigma_sec, degree)
        print('\n Times {} of sigma optimization,'.format(i))
        print(
            'val_error:{}; train_error:{}; C:{}; sigma:{};'.format(best1['val_error'], best1['train_error'], best1['C'],
                                                                   best1['sigma']))
        print('sigma_sec:{}'.format(sigma_sec))

    print('\n After Improve, ')
    print(best1)

    # 通过poly_svm.set_params可设定模型的C和gamma值
    poly_svm.set_params(degree=degree)
    poly_svm.set_params(C=best1['C'])
    poly_svm.set_params(gamma=best1['gamma'])
    poly_svm.fit(X, y)
    # 绘制决策边界
    plt.title('采取{}维多项式函数作为核函数的决策边界'.format(degree))
    plot(np.c_[X, y])
    visualize_boundary(X, poly_svm)
    plt.show(block=True)




if __name__ == '__main__':
    main()

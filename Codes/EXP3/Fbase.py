# 导入第三方库
import scipy.misc, scipy.io, scipy.optimize
from sklearn import svm
from sklearn import model_selection
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 绘制数据集
def plot(data):
    positives = data[data[:, 2] == 1]
    negatives = data[data[:, 2] == 0]

    plt.plot(positives[:, 0], positives[:, 1], 'b+')
    plt.plot(negatives[:, 0], negatives[:, 1], 'yo')


# 绘制SVM决策边界
def visualize_boundary(X, trained_svm):
    kernel = trained_svm.get_params()['kernel']
    if kernel == 'linear':
        w = trained_svm.coef_[0]
        i = trained_svm.intercept_
        xp = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
        a = -w[0] / w[1]
        b = i[0] / w[1]
        yp = a * xp - b
        plt.plot(xp, yp, 'b-')
    elif kernel == 'rbf':
        x1plot = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
        x2plot = np.linspace(min(X[:, 1]), max(X[:, 1]), 100)

        X1, X2 = np.meshgrid(x1plot, x2plot)
        vals = np.zeros(np.shape(X1))

        for i in range(0, np.shape(X1)[1]):
            this_X = np.c_[X1[:, i], X2[:, i]]
            vals[:, i] = trained_svm.predict(this_X)

        plt.contour(X1, X2, vals, colors='blue')
    elif kernel == 'poly':
        x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
        x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
        x1plot = np.linspace(x1_min, x1_max, 100)
        x2plot = np.linspace(x2_min, x2_max, 100)

        X1, X2 = np.meshgrid(x1plot, x2plot)
        vals = np.zeros(np.shape(X1))

        for i in range(0, np.shape(X1)[1]):
            this_X = np.c_[X1[:, i], X2[:, i]]
            vals[:, i] = trained_svm.predict(this_X)

        plt.contour(X1, X2, vals, colors='red')

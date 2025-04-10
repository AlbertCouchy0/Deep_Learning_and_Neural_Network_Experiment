from sklearn import svm
from sklearn import model_selection
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def find_smallest_two(lst):
    if len(lst) < 2:
        return lst, list(range(len(lst)))
    min1, min2 = (lst[0], 0), (lst[1], 1)
    if min1[0] > min2[0]:
        min1, min2 = min2, min1
    for i in range(2, len(lst)):
        if lst[i] < min1[0]:
            min2 = min1
            min1 = (lst[i], i)
        elif lst[i] < min2[0]:
            min2 = (lst[i], i)
    return [min1[0], min2[0]], [min1[1], min2[1]]

################# 基础版 #################
def params_search(X, y, X_val, y_val):
    np.c_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    sigma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    best = {'C': 0.0, 'sigma': 0.0, 'val_error': 999, 'train_error': 999}

    raveled_y = y.ravel()
    m_val = np.shape(X_val)[0]
    m = np.shape(X)[0]

    rbf_svm = svm.SVC(kernel='rbf')

    for C in np.c_values:
        for sigma in sigma_values:
            # train the SVM first
            rbf_svm.set_params(C=C)
            rbf_svm.set_params(gamma=1.0 / sigma)
            rbf_svm.fit(X, raveled_y)

            # 计算验证集错误率
            predictions = []
            for i in range(0, m_val):
                prediction_result = rbf_svm.predict(X_val[i].reshape(-1, 2))
                predictions.append(prediction_result[0])
            predictions = np.array(predictions).reshape(m_val, 1)
            val_error = (predictions != y_val.reshape(m_val, 1)).mean()

            # 计算训练集错误率
            predictions = []
            for i in range(0, m):
                prediction_result = rbf_svm.predict(X[i].reshape(-1, 2))
                predictions.append(prediction_result[0])
            predictions = np.array(predictions).reshape(m, 1)
            train_error = (predictions != y.reshape(m, 1)).mean()

            # get the lowest error
            if val_error < best['val_error']:
                best['val_error'] = val_error
                best['train_error'] = train_error
                best['C'] = C
                best['sigma'] = sigma

    best['gamma'] = 1.0 / best['sigma']
    return best

################# 二元版 #################
def params_search_dual(X, y, X_val, y_val):
    np.c_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    sigma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    best1 = {'C': 0.0, 'sigma': 0.0, 'val_error': 999, 'train_error': 999}
    best2 = {'C': 0.0, 'sigma': 0.0, 'val_error': 999, 'train_error': 999}

    raveled_y = y.ravel()
    m_val = np.shape(X_val)[0]
    m = np.shape(X)[0]
    rbf_svm = svm.SVC(kernel='rbf')

    for C in np.c_values:
        for sigma in sigma_values:
            # train the SVM first
            rbf_svm.set_params(C=C)
            rbf_svm.set_params(gamma=1.0 / sigma)
            rbf_svm.fit(X, raveled_y)

            # 计算验证集错误率
            predictions = []
            for i in range(0, m_val):
                prediction_result = rbf_svm.predict(X_val[i].reshape(-1, 2))
                predictions.append(prediction_result[0])
            predictions = np.array(predictions).reshape(m_val, 1)
            val_error = (predictions != y_val.reshape(m_val, 1)).mean()

            # 计算训练集错误率
            predictions = []
            for i in range(0, m):
                prediction_result = rbf_svm.predict(X[i].reshape(-1, 2))
                predictions.append(prediction_result[0])
            predictions = np.array(predictions).reshape(m, 1)
            train_error = (predictions != y.reshape(m, 1)).mean()

            # get the lowest error
            if val_error < best1['val_error']:
                best2['val_error'] = best1['val_error']
                best2['train_error'] = best1['train_error']
                best2['C'] = best1['C']
                best2['sigma'] = best1['sigma']
                best1['val_error'] = val_error
                best1['train_error'] = train_error
                best1['C'] = C
                best1['sigma'] = sigma

    best1['gamma'] = 1.0 / best1['sigma']
    best2['gamma'] = 1.0 / best2['sigma']
    return best1, best2

#############
def train(X, y, X_val, y_val, C, sigma):
    raveled_y = y.ravel()
    m_val = np.shape(X_val)[0]
    m = np.shape(X)[0]

    # train the SVM first
    rbf_svm = svm.SVC(kernel='rbf')
    rbf_svm.set_params(C=C)
    rbf_svm.set_params(gamma=1.0 / sigma)
    rbf_svm.fit(X, raveled_y)

    # test it out on validation data
    predictions = []
    for i in range(0, m_val):
        prediction_result = rbf_svm.predict(X_val[i].reshape(-1, 2))
        predictions.append(prediction_result[0])
    predictions = np.array(predictions).reshape(m_val, 1)
    val_error = (predictions != y_val.reshape(m_val, 1)).mean()

    # test it out on validation data
    predictions = []
    for i in range(0, m):
        prediction_result = rbf_svm.predict(X[i].reshape(-1, 2))
        predictions.append(prediction_result[0])
    predictions = np.array(predictions).reshape(m, 1)
    train_error = (predictions != y.reshape(m, 1)).mean()


    return val_error, train_error


def params_search_C(X, y, X_val, y_val, last_best1, last_C_sec):
    c1 = last_best1['C']
    c2 = last_C_sec
    if c1 > c2:
        temp = c1
        c1 = c2
        c2 = temp
    sigma = last_best1['sigma']

    np.c_values = [c1, (c1 + c2) / 2.0, c2]
    best1 = last_best1
    val_error = [0, 0, 0]
    train_error = [0, 0, 0]

    for i in range(3):
        val_error[i], train_error[i] = train(X, y, X_val, y_val, np.c_values[i], sigma)

    [error1, error2], [i1, i2] = find_smallest_two(val_error)

    best1['val_error'] = error1
    best1['train_error'] = train_error[i1]
    best1['C'] = np.c_values [i1]
    C_sec = np.c_values [i2]

    return best1, C_sec

def params_search_sigma(X, y, X_val, y_val, last_best1, last_sigma_sec):
    s1 = last_best1['sigma']
    s2 = last_sigma_sec
    if s1 > s2:
        temp = s1
        s1 = s2
        s2 = temp
    C = last_best1['C']

    sigma_values = [s1, (s1 + s2) / 2.0, s2]
    best1 = last_best1
    train_error = [0, 0, 0]
    val_error = [0, 0, 0]

    for i in range(3):
        val_error[i], train_error[i] = train(X, y, X_val, y_val, C, sigma_values[i])

    [error1, error2], [i1, i2] = find_smallest_two(val_error)

    best1['val_error'] = error1
    best1['train_error'] = train_error[i1]
    best1['sigma'] = sigma_values[i1]
    best1['gamma'] = 1.0 / best1['sigma']
    sigma_sec = sigma_values[i2]

    return best1, sigma_sec
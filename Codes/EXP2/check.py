import tensorflow as tf
from tensorflow import keras
from keras.datasets import cifar10

# 加载数据集
[(train_data, train_label), (test_data, test_label) ]= cifar10.load_data()

# 数据预处理
x_data = train_data.astype('float32') / 255.
y_data = test_data.astype('float32') / 255.

# 标签预处理
import numpy as np
def one_hot(label, num_classes):
 label_one_hot = np.eye(num_classes)[label]
 return label_one_hot
num_classes = 10 # 向量长度为10
train_label = train_label.astype('int32') # 实现变量类型转换
train_label = np.squeeze(train_label) # 去掉矩阵里维度为 1 的维度
x_label = one_hot(train_label, num_classes)
test_label = test_label.astype('int32')
y_label = np.squeeze(test_label)

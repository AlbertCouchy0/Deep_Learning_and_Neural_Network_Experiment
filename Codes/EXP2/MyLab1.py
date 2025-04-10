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

########### 构建网络 ###########
from keras import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dense, Flatten, Dropout
cnn = Sequential()
#unit1
# 二维卷积层 和 二维最大池化层 交替堆叠
cnn.add(Convolution2D(32, kernel_size=[3, 3], input_shape=(32, 32, 3), activation='relu', padding='same'))
cnn.add(Convolution2D(32, kernel_size=[3, 3], activation='relu', padding='same'))
cnn.add(MaxPooling2D(pool_size=[2, 2], padding='same'))
cnn.add(Dropout(0.5))
#unit2
# 编写网络的第二部分，可自行尝试增加更多的卷积层，改变通道数、激活函数等
# your code here#
cnn.add(Convolution2D(64, kernel_size=[3, 3], activation='relu', padding='same'))
cnn.add(Convolution2D(64, kernel_size=[3, 3], activation='relu', padding='same'))
cnn.add(MaxPooling2D(pool_size=[2, 2], padding='same'))
cnn.add(Dropout(0.5))
# 展平层
cnn.add(Flatten())
# 全连接层
cnn.add(Dense(512, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(128, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(10, activation='softmax'))

########### 编译模型 ###########
cnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['acc'])

########### 训练模型 ###########
history_cnn = cnn.fit(x_data, x_label, epochs=50, batch_size=32, shuffle=True, verbose=1, validation_split=0.1)

# 绘制图像
import matplotlib.pyplot as plt
plt.figure(1)# 损失图
plt.plot(np.array(history_cnn.history['loss']))
plt.plot(np.array(history_cnn.history['val_loss']))
plt.xlabel('Epoch')
plt.ylabel('Train loss')
plt.legend(['loss', 'val_loss'])
plt.show()
plt.figure(2)# 精度图
plt.plot(np.array(history_cnn.history['acc']))
plt.plot(np.array(history_cnn.history['val_acc']))
plt.xlabel('Epoch')
plt.ylabel('Train acc')
plt.legend(['acc', 'val_acc'])
plt.show()

test_out = cnn.predict(y_data)

# 计算准确率
num = 0
total_num = y_data.shape[0]
for i in range(total_num):
 predict = np.argmax(test_out[i])
 if predict == y_label[i]:
  num = num + 1
accuracy = num / total_num


print(accuracy)


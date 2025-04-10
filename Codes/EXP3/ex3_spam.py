# -*- coding: utf-8 -*-

import csv
import re
import pickle

import numpy as np
import nltk, nltk.stem.porter
import scipy.misc, scipy.io, scipy.optimize
from sklearn import svm


def vocaburary_mapping():
	vocab_list = {}
	with open('vocab.txt', 'r') as file:
		reader = csv.reader(file, delimiter='\t')
		for row in reader:
			vocab_list[row[1]] = int(row[0])

	return vocab_list

def feature_extraction(word_indices):
	features = np.zeros((1899, 1))
	for index in word_indices:
		features[index] = 1
	return features


def email_preprocess(email):
	with open(email, 'r') as f:
		email_contents = f.read()
	vocab_list = vocaburary_mapping()
	word_indices = []
	email_contents = email_contents.lower()
	email_contents = re.sub('<[^<>]+>', ' ', email_contents)
	email_contents = re.sub('[0-9]+', 'number', email_contents)
	email_contents = re.sub('(http|https)://[^\s]*', 'httpaddr', email_contents)
	email_contents = re.sub('[^\s]+@[^\s]+', 'emailaddr', email_contents)
	email_contents = re.sub('[$]+', 'dollar', email_contents)
	stemmer = nltk.stem.porter.PorterStemmer()
	tokens = re.split('[ ' + re.escape("@$/#.-:&*+=[]?!(){},'\">_<;%\n") + ']', email_contents)

	for token in tokens:
		token = re.sub('[^a-zA-Z0-9]', '', token)
		token = stemmer.stem(token.strip())

		if len(token) == 0:
			continue

		if token in vocab_list:
			word_indices.append(vocab_list[token])

	return word_indices, ' '.join(tokens)



# 预处理
def part_1():
	word_indices, processed_contents = email_preprocess('emailSample1.txt')
	print(word_indices)
	print(processed_contents)


# 特征提取
def part_2():
	word_indices, processed_contents = email_preprocess('emailSample1.txt')
	features = feature_extraction(word_indices)
	print(features)


# SVM模型训练
def part_3():
	#加载训练集
	mat = scipy.io.loadmat("spamTrain.mat")
	X, y = mat['X'], mat['y']

	# linear_svm = pickle.load(open("linear_svm.svm", "rb")) # 模型加载
	#训练SVM
	linear_svm = svm.SVC(C=0.1, kernel='linear')
	linear_svm.fit(X, y.ravel())
	# pickle.dump(linear_svm, open("linear_svm.svm", "wb")) # 模型保存
	# 预测并计算训练集正确率
	predictions = linear_svm.predict(X)
	predictions = predictions.reshape(np.shape(predictions)[0], 1)
	print('Accuracy rate for training set:')
	print('{}%'.format((predictions == y).mean() * 100.0))
	# 加载测试集
	mat = scipy.io.loadmat("spamTest.mat")
	X_test, y_test = mat['Xtest'], mat['ytest']
	# 预测并计算测试集正确率
	predictions = linear_svm.predict(X_test)
	predictions = predictions.reshape(np.shape(predictions)[0], 1)
	print('Accuracy rate for testing set:')
	print('{}%'.format((predictions == y_test).mean() * 100.0))

	vocab_list = vocaburary_mapping()
	reversed_vocab_list = dict((v, k) for (k, v) in vocab_list.items())
	sorted_indices = np.argsort(linear_svm.coef_, axis=None)

	for i in sorted_indices[0:15]:
		print(reversed_vocab_list[i])

def part_4():
	mat = scipy.io.loadmat("spamTrain.mat")
	X, y = mat['X'], mat['y']

	# linear_svm = pickle.load(open("linear_svm.svm", "rb"))
	linear_svm = svm.SVC(C=0.1, kernel='linear')
	linear_svm.fit(X, y.ravel())
	# pickle.dump(linear_svm, open("linear_svm.svm", "wb"))

	word_indices1, _ = email_preprocess('spamSample1.txt')
	features1 = feature_extraction(word_indices1).transpose()
	print(linear_svm.predict(features1))

	word_indices2, _ = email_preprocess('spamSample2.txt')
	features2 = feature_extraction(word_indices2).transpose()
	print(linear_svm.predict(features2))


part_1()
part_2()
part_3()
part_4()
# print(vocaburary_mapping())

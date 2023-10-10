# -*- coding: utf-8 -*-
import numpy as np
from pandas import read_csv
import pandas as pd
import matplotlib
from sklearn.preprocessing import OneHotEncoder
from sympy import *

X_train = read_csv(r'E:\Deep Learning\Data\train_in.csv') #(1706, 256)
y_train = read_csv(r'E:\Deep Learning\Data\train_out.csv') #(1706, 1)
X_test = read_csv(r'E:\Deep Learning\Data\test_in.csv') #(999, 256)
y_test = read_csv(r'E:\Deep Learning\Data\test_out.csv') #(999, 1)

y_train.columns = ['class']
# Append a column of ones for bias to X_train and X_test 添加偏置项
X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1)))) #水平方向叠加偏置项 (1706, 257)
X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1))))

'''activation function'''
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z)) # z is the output of the nerual netwok


def sigmoid_prime(z): #The derivative of the sigmoid function
    return sigmoid(z) * (1 - sigmoid(z))

'''loss function'''
def cross_entropy(true_distribution, predicted_distribution):
    cross_entropy = -np.sum(np.multiply(true_distribution, 
                                        np.log(predicted_distribution)))
    
    return cross_entropy



#Weight initialization 权重W初始化
def initialize_weights(input_size, output_size):
    np.random.seed(0)  #固定训练结果，使之可复现
    return np.random.rand(input_size, output_size) #返回分布的平均值=0，方差=1

def train_perceptron(X_train, y_train, num_epochs=1000, learning_rate=0.1):
    input_size, hidden_size, output_size = X_train.shape[1], 16, 10 #256+1 inputs,
    W = initialize_weights(input_size, hidden_size) #(257, 16)
    V = initialize_weights(hidden_size, output_size) #(16, 10)
    
    for epoch in range(num_epochs):
        # Compute scores for all samples in one batch using matrix multiplication
        #矩阵乘法，完成前向传播
        scores = np.dot(X_train, W)
        scores_sigmod = sigmoid(scores) #(1706, 16)
        y_predict = np.dot(scores_sigmod,V) #(1706,10)
        
        # Predict the classes for all samples 预测每个X对应的数字
        predicted_classes = np.argmax(y_predict, axis=1).reshape(1706,1)
        # Find misclassified samples #找到不符合真实数字的X
        misclassified = np.where(predicted_classes != y_train)[0]
        
        y_train_dummies = pd.get_dummies(y_train, columns=['class'], dtype=int)
        loss = cross_entropy(y_train_dummies, y_predict)  # 
        
        # 计算梯度 gradient_v 和 gradient_w
        gradient_v = diff(loss,V)
        gradient_w = diff(loss,W)
        #dy = y_predict - y_train) * sigmoid_prime(y_predict) / y_train.shape[1 # 损失函数关于网络输出的梯度
        if misclassified.shape[0] == 0:
            break
        
        # Update weights for misclassified samples using matrix operations
        #反向传播，更新W
        W -= learning_rate * gradient_w
        V -= learning_rate * gradient_v
    
    return W

def evaluate_perceptron(W, X):
    # Compute the scores for all samples using matrix multiplication
    scores = np.dot(X_train, W)
    scores_sigmod = sigmoid(scores) #(1706, 16)
    y_predict = np.dot(scores_sigmod,V)
    
    # Predict the classes
    predicted_classes = np.argmax(y_predict, axis=1).reshape(1706,1)
    # Find misclassified samples #找到不符合真实数字的X
    misclassified = np.where(predicted_classes != y_train)[0]
    accuracy = (y_predict.shape[0]-len(misclassified)) / y_predict.shape[0]
    
    y_train_dummies = pd.get_dummies(y_train, columns=['class'], dtype=int)
    loss = cross_entropy(y_train_dummies, y_predict)
    
    return accuracy,loss


def mlp():
    num_experiments = 5
    train_accuracies = []
    test_accuracies = []
    train_loss = []
    test_loss = []
    
    for _ in range(num_experiments):
        # Training the perceptron
        W = train_perceptron(X_train, y_train)
        
        # Evaluating on the train set
        accuracy1,loss1 = evaluate_perceptron(W, X_train).reshape(1706,1)
        train_accuracies.append(accuracy1)
        train_loss.append(loss1)
        
        # Evaluating on the test set
        accuracy2, loss2 = evaluate_perceptron(W, X_test).reshape(999,1)
        test_accuracies.append(accuracy2)
        test_loss.append(loss2)
    
    avg_train_accuracy = np.mean(train_accuracies)
    avg_test_accuracy = np.mean(test_accuracies)
    
    print(f"Average Train Accuracy: {avg_train_accuracy * 100:.2f}%")
    print(f"Average Test Accuracy: {avg_test_accuracy * 100:.2f}%")

mlp()
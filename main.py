#!/usr/bin/env python
# -*- coding:utf-8 -*-
# file:main.py
# author:lizhihao
# datetime:2024-02-06 16:27
# software: PyCharm


import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import TransformerModel
import model_self
def pre_data():
    data_frames=[]
    for i in range(1,31):
        file_name=f'./data/{i}.csv'
        data = pd.read_csv(file_name, parse_dates={'datetime': ['date', 'time']}, index_col='datetime')
        training_data_len = math.ceil(len(data) * .9)
        train_data = data[:training_data_len]
        test_data = data[training_data_len:]
        # scaler = MinMaxScaler(feature_range=(0, 1))
        # scaled_train = scaler.fit_transform(train_data)
        # scaled_test = scaler.fit_transform(test_data)
        scaled_train = train_data.values
        scaled_test = test_data.values
        sequence_length = 30
        X_train, y_train = [], []
        for i in range(len(scaled_train) - sequence_length):
            X_train.append(scaled_train[i:i + sequence_length])
        y_train = scaled_train[sequence_length:, -1]
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        sequence_length = 30
        X_test, y_test = [], []
        for i in range(len(scaled_test) - sequence_length):
            X_test.append(scaled_test[i:i + sequence_length])
        y_test = scaled_test[sequence_length:, -1]
        X_test, y_test = np.array(X_test), np.array(y_test)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)
        data1={}
        data1["X_train"]=X_train
        data1["y_train"]=y_train
        data1["X_test"]=X_test
        data1["y_test"]=y_test
        data_frames.append(data1)
    return data_frames

def pre_train(data,model,device,criterion,optimizer):
    batch_size = 32
    num_epochs = 10  # 训练轮数
    for i in range(0,30):
        print(i)
        for epoch in range(num_epochs):
            model.train()
            for i in range(0, len(data[i]["X_train"]), batch_size):
                # 获取批次数据
                inputs = data[i]["X_train"][i:i + batch_size].to(device)
                labels = data[i]["y_train"][i:i + batch_size].to(device)
                # 前向传播
                outputs = model(inputs,[0,0,0,0])
                loss = criterion(outputs.squeeze(), labels)
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')



def main_():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data=pre_data()
    # 模型参数
    input_dim = 7  # 输入特征的维度
    model_dim = 64  # Transformer模型的维度
    num_heads = 2  # 多头注意力机制的头数
    num_layers = 4  # 编码器层的数量
    output_dim = 1  # 输出维度，预测一个标签值
    # 初始化模型
    model = model_self.TransformerModel(input_dim, model_dim, num_heads, num_layers, output_dim).to(device)
    # 定义损失函数和优化器
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 训练数据
    # 假设X_train和y_train已经是正确的形状：X_train [30038, 30, 7], y_train [30038]
    # 由于代码限制，这里不再演示数据加载部分
    pre_train(data,model,device,criterion,optimizer)
    for layer in enumerate(model.transformer_encoder.layers):



    model.eval()
    predictions = []
    with torch.no_grad():  # 在不计算梯度的情况下执行前向传播，以节省内存并加速
        for i in range(len(X_test)):
            inputs = X_test[i:i + 1].to(device)  # 获取单个样本进行预测
            output = model(inputs)
            predictions.append(output.cpu().item())  # 存储预测结果
    predictions_tensor = torch.tensor(predictions, dtype=torch.float32)
    # 计算均方误差
    mse = criterion(predictions_tensor, y_test)
    print(f'Test L1loss: {mse.item()}')
    # 将y_test转换为列表以便绘图
    true_values = y_test.tolist()
    plt.figure(figsize=(12, 6))  # 设置图形的大小
    plt.plot(true_values, label='True Values', color='blue', marker='o', linestyle='-', markersize=1)  # 真实值折线图
    plt.plot(predictions, label='Predictions', color='red', linestyle='-', linewidth=1)  # 预测值折线图
    plt.title('True Values vs Predictions')  # 图形标题
    plt.xlabel('Sample Index')  # x轴标签
    plt.ylabel('Value')  # y轴标签
    plt.legend()  # 显示图例
    plt.show()


if __name__ == '__main__':
    main_()
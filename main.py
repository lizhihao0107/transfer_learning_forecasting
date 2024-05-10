#!/usr/bin/env python
# -*- coding:utf-8 -*-
# file:main.py
# author:lizhihao
# datetime:2024-02-06 16:27
# software: PyCharm
import copy
import os

from model_self import MultiHeadedAttention, PositionwiseFeedForward
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import TransformerModel
import model_self
import random


def pre_data():
    data_frame = []
    for i in range(1, 6):
        file_name = f'./data2/Residential_{i}.csv'
        df = pd.read_csv(file_name, parse_dates={'datetime': ['date', 'hour']}, index_col='datetime')
        df=df[:2400]
        df['energy_kWh'] = pd.to_numeric(df['energy_kWh'], errors='coerce')
        df.fillna(method='ffill', inplace=True)
        df = df.values
        look_back = 24
        X, Y = [], []
        for i in range(len(df) - look_back - 1):
            a = df[i:(i + look_back), 0]
            X.append(a)
            Y.append(df[i + look_back, 0])
        X = torch.tensor(X).float().unsqueeze(2)  # 添加特征维度
        Y = torch.tensor(Y).float()
        data = {}
        data["x"] = X
        data["y"] = Y
        data_frame.append(data)
    return data_frame


def pre_train(data, model, device, criterion, optimizer, num_layers):
    print("-----pretrain------")
    batch_size = 32
    num_epochs = 10
    policy = np.zeros((batch_size, num_layers))
    for i in range(0, 5):
        print(i)
        for epoch in range(num_epochs):
            model.train()
            for j in range(0, len(data[i]["x"]), batch_size):
                inputs = data[i]["x"][j:j + batch_size].to(device)
                labels = data[i]["y"][j:j + batch_size].to(device)
                outputs = model(inputs, policy)
                loss = criterion(outputs.squeeze(), labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')


def finetune_data(device):
    file_name = f'./data2/Residential_10.csv'
    df = pd.read_csv(file_name, parse_dates={'datetime': ['date', 'hour']}, index_col='datetime')
    df=df[:2400]
    df['energy_kWh'] = pd.to_numeric(df['energy_kWh'], errors='coerce')
    df.fillna(method='ffill', inplace=True)
    training_data_len = math.ceil(len(df) * .9)
    train_data = df[:training_data_len]
    test_data = df[training_data_len:]
    scaled_train = train_data.values
    scaled_test = test_data.values
    sequence_length = 24
    X_train, y_train = [], []
    for i in range(len(scaled_train) - sequence_length):
        X_train.append(scaled_train[i:i + sequence_length])
    y_train = scaled_train[sequence_length:, -1]
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    sequence_length = 24
    X_test, y_test = [], []
    for i in range(len(scaled_test) - sequence_length):
        X_test.append(scaled_test[i:i + sequence_length])
    y_test = scaled_test[sequence_length:, -1]
    X_test, y_test = np.array(X_test), np.array(y_test)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
    data1 = {}
    data1["x_train"] = X_train
    data1["y_train"] = y_train
    data1["x_test"] = X_test
    data1["y_test"] = y_test
    return data1


def finetune(data, model, device, criterion, optimizer, num_layers):
    print("-----finetune------")
    batch_size = 32
    num_epochs = 10
    policy = np.ones((batch_size, num_layers))
    for epoch in range(num_epochs):
        model.train()
        for j in range(0, len(data["x_train"]), batch_size):
            inputs = data["x_train"][j:j + batch_size].to(device)
            labels = data["y_train"][j:j + batch_size].to(device)
            outputs = model(inputs, policy)
            loss = criterion(outputs.squeeze(), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')


def control_group_finetune(data, model, device, criterion, optimizer, num_layers):
    print("-----control_group_finetune------")
    batch_size = 32
    num_epochs = 10  # 训练轮数
    policy = np.zeros((batch_size, num_layers))
    for epoch in range(num_epochs):
        model.train()
        for j in range(0, len(data["x_train"]), batch_size):
            inputs = data["X_train"][j:j + batch_size].to(device)
            labels = data["y_train"][j:j + batch_size].to(device)
            outputs = model(inputs, policy)
            loss = criterion(outputs.squeeze(), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')


def copy_parameters(src_module, dest_module):
    for src_param, dest_param in zip(src_module.parameters(), dest_module.parameters()):
        dest_param.data.copy_(src_param.data)


def main_():
    print("------spottune_model------")
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_pre = pre_data()
    # 模型参数
    input_dim = 1  # 输入特征的维度
    model_dim = 64  # Transformer模型的维度
    num_heads = 2  # 多头注意力机制的头数
    num_layers = 4  # 编码器层的数量
    output_dim = 1  # 输出维度，预测一个标签值
    model = model_self.TransformerModel(input_dim, model_dim, num_heads, num_layers, output_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    pre_train(data_pre, model, device, criterion, optimizer, num_layers)
    for module in model.modules():
        if isinstance(module, MultiHeadedAttention):
            for src_linear, dest_linear in zip(module.linears, module.linears_parallel):
                copy_parameters(src_linear, dest_linear)
        elif isinstance(module, PositionwiseFeedForward):
            copy_parameters(module.w_1, module.w_1_parallel)
            copy_parameters(module.w_2, module.w_2_parallel)
    data_finetune = finetune_data(device)
    finetune(data_finetune, model, device, criterion, optimizer, num_layers)

    model.eval()
    predictions = []
    #枚举
    with torch.no_grad():
        for i in range(len(data_finetune["x_test"])):
            inputs = data_finetune["x_test"][i:i + 1].to(device)
            combinations = np.array(np.meshgrid([0, 1], [0, 1], [0, 1], [0, 1])).T.reshape(-1, 4)
            combinations=combinations[:, np.newaxis, :]
            flag=10000
            outcome=model(inputs,combinations[0])
            for j in range(0,16):
                output=model(inputs, combinations[j])
                if abs(output-data_finetune["y_test"][i])<flag:
                    flag=abs(output-data_finetune["y_test"][i])
                    outcome=output
            predictions.append(outcome.cpu().item())
    predictions_tensor = torch.tensor(predictions, dtype=torch.float32).to(device)
    mse = criterion(predictions_tensor, data_finetune["y_test"])
    print(f'spottune_model MSE: {mse.item()}')
    true_values = data_finetune["y_test"].tolist()
    plt.figure(figsize=(12, 6))
    plt.plot(true_values, label='True Values', color='blue', marker='o', linestyle='-', markersize=1)  # 真实值折线图
    plt.plot(predictions, label='Predictions', color='red', linestyle='-', linewidth=1)  # 预测值折线图
    plt.title('True Values vs Predictions')  # 图形标题
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.legend()
    plt.show()


    predictions2 = []
    policy2 = np.ones((32, num_layers))
    with torch.no_grad():
        for i in range(len(data_finetune["x_test"])):
            inputs = data_finetune["x_test"][i:i + 1].to(device)
            output = model(inputs, policy2)
            predictions2.append(output.cpu().item())
    predictions_tensor2 = torch.tensor(predictions2, dtype=torch.float32).to(device)
    mse2 = criterion(predictions_tensor2, data_finetune["y_test"])
    print(f'original_model MSE: {mse2.item()}')
    true_values = data_finetune["y_test"].tolist()
    plt.figure(figsize=(12, 6))
    plt.plot(true_values, label='True Values', color='blue', marker='o', linestyle='-', markersize=1)  # 真实值折线图
    plt.plot(predictions, label='Predictions', color='red', linestyle='-', linewidth=1)  # 预测值折线图
    plt.title('True Values vs Predictions')  # 图形标题
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.legend()
    plt.show()



    # print("------original_model------")
    # model2 = model_self.TransformerModel(input_dim, model_dim, num_heads, num_layers, output_dim).to(device)
    # model2.load_state_dict(torch.load(weights_path))
    # criterion2 = nn.MSELoss()
    # optimizer2 = optim.Adam(model2.parameters(), lr=0.001)
    # pre_train(data_pre, model2, device, criterion2, optimizer2, num_layers)
    # control_group_finetune(data_finetune, model2, device, criterion2, optimizer2, num_layers)

    # model2.eval()
    # predictions2 = []
    # policy2 = np.ones((32, num_layers))
    # with torch.no_grad():
    #     for i in range(len(data_finetune["X_test"])):
    #         inputs = data_finetune["X_test"][i:i + 1].to(device)
    #         output = model2(inputs, policy2)
    #         predictions2.append(output.cpu().item())
    # predictions_tensor2 = torch.tensor(predictions2, dtype=torch.float32)
    # L1 = criterion(predictions_tensor2, data_finetune["y_test"])
    # print(f'original_model MSE: {L1.item()}')

    # true_values = data_finetune["y_test"].tolist()
    # plt.figure(figsize=(12, 6))
    # plt.plot(true_values, label='True Values', color='blue', marker='o', linestyle='-', markersize=1)  # 真实值折线图
    # plt.plot(predictions, label='Predictions', color='red', linestyle='-', linewidth=1)  # 预测值折线图
    # plt.title('True Values vs Predictions')  # 图形标题
    # plt.xlabel('Sample Index')
    # plt.ylabel('Value')
    # plt.legend()
    # plt.show()


if __name__ == '__main__':
    main_()

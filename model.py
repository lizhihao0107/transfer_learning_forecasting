#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:model.py
# author:lizhihao
# datetime:2024-02-07 17:22
# software: PyCharm
import math
import torch
import torch.nn as nn
import torch.optim as optim

class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, dropout=0.1, max_len=30):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2) * -(math.log(10000.0) / model_dim))
        pe = torch.zeros(max_len, 1, model_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim, max_len=5000):
        super(TransformerModel, self).__init__()
        self.model_dim = model_dim
        self.input_linear = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_linear = nn.Linear(model_dim, output_dim)

    def forward(self, src):
        src = self.input_linear(src)  # [batch_size, seq_length, input_dim] -> [batch_size, seq_length, model_dim]
        src = src.permute(1, 0, 2)  # 调整维度以符合Transformer的输入要求：[seq_length, batch_size, model_dim]
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output.permute(1, 0, 2)  # 转换回原来的维度
        output = self.output_linear(output[:, -1, :])  # 只取序列的最后一个时间步的输出用于预测
        return output


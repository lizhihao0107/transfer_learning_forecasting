#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:model_self.py
# author:lizhihao
# datetime:2024-02-15 0:28
# software: PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy


def attention(query, key, value, mask=None, dropout=None):
    # query, key, value的形状类似于(30, 8, 10, 64), (30, 8, 11, 64),
    # (30, 8, 11, 64)，例如30是batch.size，即当前batch中有多少一个序列；
    # 8=head.num，注意力头的个数；
    # 10=目标序列中词的个数，64是每个词对应的向量表示；
    # 11=源语言序列传过来的memory中，当前序列的词的个数，
    # 64是每个词对应的向量表示。
    # 类似于，这里假定query来自target language sequence；
    # key和value都来自source language sequence.
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)  # 64=d_k
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # 先是(30,8,10,64)和(30, 8, 64, 11)相乘，
    # （注意是最后两个维度相乘）得到(30,8,10,11)，
    # 代表10个目标语言序列中每个词和11个源语言序列的分别的“亲密度”。
    # 然后除以sqrt(d_k)=8，防止过大的亲密度。
    # 这里的scores的shape是(30, 8, 10, 11)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
        # 使用mask，对已经计算好的scores，按照mask矩阵，填-1e9，
        # 然后在下一步计算softmax的时候，被设置成-1e9的数对应的值~0,被忽视
    p_attn = F.softmax(scores, dim=-1)
    # 对scores的最后一个维度执行softmax，得到的还是一个tensor,(30, 8, 10, 11)
    if dropout is not None:
        p_attn = dropout(p_attn)  # 执行一次dropout
    return torch.matmul(p_attn, value), p_attn


# 返回的第一项，是(30,8,10, 11)乘以（最后两个维度相乘）
# value=(30,8,11,64)，得到的tensor是(30,8,10,64)，
# 和query的最初的形状一样。另外，返回p_attn，形状为(30,8,10,11).
# 注意，这里返回p_attn主要是用来可视化显示多头注意力机制。

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        # h=8, d_model=512
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0  # We assume d_v always equals d_k 512%8=0
        self.d_k = d_model // h  # d_k=512//8=64
        self.h = h  # 8
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.linears_parallel = clones(nn.Linear(d_model, d_model), 4)
        # 定义四个Linear networks, 每个的大小是(512, 512)的，
        # 每个Linear network里面有两类可训练参数，Weights，其大小为512*512，以及biases，其大小为512=d_model。
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, action, mask=None):
        # 注意，输入query的形状类似于(30, 10, 512)，
        # key.size() ~ (30, 11, 512),
        # 以及value.size() ~ (30, 11, 512)
        nbatches = query.size(0)  # e.g., nbatches=30
        if action == 0:
            query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k)
                                     .transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]
            x, self.attn = attention(query, key, value, mask=mask,
                                     dropout=self.dropout)
            x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
            return self.linears[-1](x)
        elif action == 1:
            query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k)
                                     .transpose(1, 2) for l, x in zip(self.linears_parallel, (query, key, value))]
            x, self.attn = attention(query, key, value, mask=mask,
                                     dropout=self.dropout)
            x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
            return self.linears_parallel[-1](x)


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        # features=d_model=512, eps=epsilon 用于分母的非0化平滑
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        # a_2 是一个可训练参数向量，(512)
        self.b_2 = nn.Parameter(torch.zeros(features))
        # b_2 也是一个可训练参数向量, (512)
        self.eps = eps

    def forward(self, x):
        # x 的形状为(batch.size, sequence.len, 512)
        mean = x.mean(-1, keepdim=True)
        # 对x的最后一个维度，取平均值，得到tensor (batch.size, seq.len)
        std = x.std(-1, keepdim=True)
        # 对x的最后一个维度，取标准方差，得(batch.size, seq.len)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        # 本质上类似于（x-mean)/std，不过这里加入了两个可训练向量
        # a_2 and b_2，以及分母上增加一个极小值epsilon，用来防止std为0的时候的除法溢出


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        # size=d_model=512; dropout=0.1
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)  # (512)，用来定义a_2和b_2
        self.dropout = nn.Dropout(dropout)
        self.flag = 0

    def forward(self, x, sublayer, *action):
        "Apply residual connection to any sublayer with the "
        "same size."
        # x is alike (batch.size, sequence.len, 512)
        # sublayer是一个具体的MultiHeadAttention
        # 或者PositionwiseFeedForward对象
        if self.flag == 0:
            return x + self.dropout(sublayer(self.norm(x)))
        elif self.flag == 1:
            return x + self.dropout(sublayer(self.norm(x), action[0]))
        # x (30, 10, 512) -> norm (LayerNorm) -> (30, 10, 512)
        # -> sublayer (MultiHeadAttention or PositionwiseFeedForward)-> (30, 10, 512) -> dropout -> (30, 10, 512)
        # 然后输入的x（没有走sublayer) + 上面的结果，
        # 即实现了残差相加的功能


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        # d_model = 512
        # d_ff = 2048 = 512*4
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_1_parallel = nn.Linear(d_model, d_ff)
        # 构建第一个全连接层，(512, 2048)，其中有两种可训练参数：
        # weights矩阵，(512, 2048)，以及
        # biases偏移向量, (2048)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.w_2_parallel = nn.Linear(d_ff, d_model)
        # 构建第二个全连接层, (2048, 512)，两种可训练参数：
        # weights矩阵，(2048, 512)，以及
        # biases偏移向量, (512)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, action):
        # x shape = (batch.size, sequence.len, 512)
        # 例如, (30, 10, 512)
        if action == 0:
            return self.w_2(self.dropout(F.relu(self.w_1(x))))
        elif action == 1:
            return self.w_2_parallel(self.dropout(F.relu(self.w_1_parallel(x))))
        # x (30, 10, 512) -> self.w_1 -> (30, 10, 2048)-> relu -> (30, 10, 2048)-> dropout -> (30, 10, 2048)
        # -> self.w_2 -> (30, 10, 512)是输出的shape


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and "
    "feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout=0.1):
        # size=d_model=512
        # self_attn = MultiHeadAttention对象, first sublayer
        # feed_forward = PositionwiseFeedForward对象，second sublayer
        # dropout = 0.1 (e.g.)
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        # 使用深度克隆方法，完整地复制出来两个SublayerConnection
        self.sublayer[0].flag = 0
        self.sublayer[1].flag = 1
        self.size = size  # 512

    def forward(self, x, action):
        "Follow Figure 1 (left) for connections."
        # x shape = (30, 10, 512)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, action))
        # x (30, 10, 512) -> self_attn (MultiHeadAttention)
        # shape is same (30, 10, 512) -> SublayerConnection-> (30, 10, 512)
        return self.sublayer[1](x, self.feed_forward, action)
        # x 和feed_forward对象一起，给第二个SublayerConnection


class Encoder(nn.Module):
    def __init__(self, layer, N):
        # layer = one EncoderLayer object, N=6
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        # 深copy，N=6，
        self.norm = LayerNorm(layer.size)
        # 定义一个LayerNorm，layer.size=d_model=512
        # 其中有两个可训练参数a_2和b_2

    def forward(self, x, policy):
        "Pass the input (and mask) through each layer in turn."
        # x is alike (30, 10, 512)
        # (batch.size, sequence.len, d_model)
        for i, layer in enumerate(self.layers):
            x = layer(x, policy[i])
            # 进行六次EncoderLayer操作
        return self.norm(x)
        # 最后做一次LayerNorm，最后的输出也是(30, 10, 512) shape


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
        self.self_attn = MultiHeadedAttention(num_heads, model_dim)
        self.feed_forward = PositionwiseFeedForward(model_dim, model_dim * 4)
        self.encoder_layer = EncoderLayer(model_dim, self.self_attn, self.feed_forward)
        self.transformer_encoder = Encoder(self.encoder_layer, num_layers)
        self.output_linear = nn.Linear(model_dim, output_dim)

    def forward(self, src, policy):
        src = self.input_linear(src)  # [batch_size, seq_length, input_dim] -> [batch_size, seq_length, model_dim]
        src = src.permute(1, 0, 2)  # 调整维度以符合Transformer的输入要求：[seq_length, batch_size, model_dim]
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, policy)
        output = output.permute(1, 0, 2)  # 转换回原来的维度
        print("output1")
        print(output[:, -1, :].shape)
        output = self.output_linear(output[:, -1, :])  # 只取序列的最后一个时间步的输出用于预测
        print("output2")
        print(output.shape)
        print("output3")
        print(output.squeeze().shape)
        return output

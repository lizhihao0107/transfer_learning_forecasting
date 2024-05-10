#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:gumbel_softmax.py
# author:lizhihao
# datetime:2024-02-23 22:04
# software: PyCharm
import torch
import torch.nn.functional as F
from torch.autograd import Variable

def sample_gumbel(shape, eps=1e-20):
    U = torch.cuda.FloatTensor(shape).uniform_()
    return -Variable(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature = 1):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    # y_hard = torch.zeros_like(y).view(-1, shape[-1])
    # y_hard.scatter_(1, ind.view(-1, 1), 1)
    # y_hard = y_hard.view(*shape)
    # return (y_hard - y).detach() + y
    y_hard = F.one_hot(ind, num_classes=2).to(logits.device)
    # 取 one-hot 编码中的第二列，表示选择最大值时的索引（0或1）
    y_hard = y_hard[..., 1]
    # Straight-through estimator trick
    # 在前向传播中使用 y_hard，但在反向传播中使用 y 的梯度
    y_hard = (y_hard - y[..., 1].detach()) + y[..., 1]
    return y_hard


def soft_one_hot(logits, temperature=1.0):
    """
    input: [batch, n_class, 2]
    return: [batch, n_class] an one-hot vector
    """
    # 计算 softmax
    soft_y = F.softmax(logits / temperature, dim=-1)

    # 使用 argmax 选择最大 logit 值的索引
    ind = torch.argmax(logits, dim=-1)

    # 转换为 one-hot 编码
    y_hard = F.one_hot(ind, num_classes=2).to(logits.device)

    # 从 one-hot 编码中选择第二列
    y_hard = y_hard[..., 1]

    # 使用直通估计器技术进行前向传播和反向传播
    y_hard = (y_hard - soft_y[..., 1].detach()) + soft_y[..., 1]

    return y_hard
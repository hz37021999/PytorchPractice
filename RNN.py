#RNN 处理具有序列连接的 如天气 股市 自然语言
#RNN cell    xt-->RNN cell-->ht 一个维度映射到另一个维度 本质是线性层
#文本 cnn+fc生成h0 然后接上rnn   同一个rnn cell反复运算

import torch

batch_size = 1
seq_len = 3
input_size = 4
hidden_size = 2

cell = torch.nn.RNNCell(input_size=input_size, hidden_size=hidden_size)

dataset = torch.randn(seq_len,batch_size,input_size)
hidden = torch.zeros(batch_size,hidden_size)

for idx, input in enumerate(dataset):
    print('='*20, idx,'='*20)
    print('Input size: ',input.shape)

    hidden = cell(input,hidden)

    print('Output size: ',hidden.shape)
    print(hidden)


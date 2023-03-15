import torch
import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dp=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(p=dp)

    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, head_num, len_q, dim_hidden]
        K: [batch_size, head_num, len_k, dim_hidden]
        V: [batch_size, head_num, len_v(=len_k), dim_hidden]
        attn_mask: [batch_size, head_num, seq_len, seq_len]
        '''
        dim_hidden = Q.shape[-1]
        # scores : [batch_size, head_num, len_q, len_k]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(dim_hidden)
        # Fills elements of self tensor with value where mask is True.
        scores.masked_fill_(attn_mask, -1e9)

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, head_num, len_q, dim_hidden]
        context = self.dropout(context)
        return context, attn
        
class MultiHeadAttention(nn.Module):
    '''
        forward : (input_Q, input_K, input_V, attn_mask)
    '''
    def __init__(self, num_hidden, head_num, dim_hidden, dp=0.1):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(num_hidden, dim_hidden * head_num, bias=False)
        self.W_K = nn.Linear(num_hidden, dim_hidden * head_num, bias=False)
        self.W_V = nn.Linear(num_hidden, dim_hidden * head_num, bias=False)
        self.fc = nn.Linear(head_num * dim_hidden, num_hidden, bias=False)
        self.head_num = head_num
        self.dim_hidden = dim_hidden
        self.dropout = nn.Dropout(p=dp)
        self.SDPA = ScaledDotProductAttention(dp)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q: [batch_size, len_q, num_hidden]
        input_K: [batch_size, len_k, num_hidden]
        input_V: [batch_size, len_v(=len_k), num_hidden]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        head_num = self.head_num
        dim_hidden = self.dim_hidden
        num_hidden = input_Q.shape[-1]

        Q = self.W_Q(input_Q).view(batch_size, -1, head_num, dim_hidden).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, head_num, dim_hidden).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, head_num, dim_hidden).transpose(1, 2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, head_num, 1, 1)

        context, attn = self.SDPA(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, head_num * dim_hidden)
        output = self.fc(context)
        output = self.dropout(output)
        return nn.LayerNorm(num_hidden).to(output.device)(output + residual), attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, num_hidden, dp=0.1):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(num_hidden, num_hidden, bias=False)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dp)
        self.fc2 = nn.Linear(num_hidden, num_hidden, bias=False)
        

    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, num_hidden]
        '''
        residual = inputs
        num_hidden = inputs.shape[-1]
        output = self.fc1(inputs)
        output = self.dropout(inputs)
        output = self.relu(inputs)
        output = self.fc2(inputs)
        # [batch_size, seq_len, num_hidden]
        return nn.LayerNorm(num_hidden).to(inputs.device)(output + residual)
import torch
import numpy as np
import sys

def get_attn_pad_mask(seq_q, seq_k, valid, blank_pad=False):
    '''
        seq_q: [batch_size, seq_len]
        seq_k: [batch_size, seq_len]
        seq_len could be src_len or it could be tgt_len
        seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    batch_size, len_q = seq_q.shape[:2]
    batch_size, len_k = seq_k.shape[:2]
    # eq(valid) is PAD token
    pad_attn_mask = seq_k.data.eq(valid)
    if blank_pad:
        num = len_k - torch.sum(pad_attn_mask, dim=-1)
        idx = [i for i in range(batch_size)]
        pad_attn_mask[idx,num] = False
        # for i in range(batch_size):
        #     for j in range(15):
        #         a = seq_k[i,j]
        #         print('{: >7}'.format(str(int(a))), end='')
        #     print('')
        #     for k in range(15):
        #         print('{: >7}'.format(str(bool(pad_attn_mask[i,k]))), end='')
        #     print('')
        # sys.exit()
    # [batch_size, len_q, len_k]
    pad_attn_mask = pad_attn_mask.unsqueeze(1)
    return pad_attn_mask.expand(batch_size, len_q, len_k)

def get_attn_subsequence_mask(seq):
    '''
        seq: [batch_size, tgt_len]
    '''
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    # Upper triangular matrix
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)
    subsequence_mask = torch.from_numpy(subsequence_mask).bool()
    return subsequence_mask

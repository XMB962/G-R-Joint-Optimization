# 计算工具：数据类型；WER
# imported by lib.core.function

import numpy as np
import math

class AverageMeter(object):
    '''含历史均值的数据类型'''
    def __init__(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def update_dict(self, val, n=1):
        def to_item(inp):
            try:
                inp = inp.item()
            except:
                pass
            return inp

        self.count += n
        self.val = val
        if not isinstance(self.sum, dict):
            self.sum = {}
            self.avg = {}
            self.display = {}
            for k,v in val.items():
                v = to_item(v)
                self.sum[k] = v * n
                self.avg[k] = self.sum[k]/self.count
                self.display[k] = [v, self.avg[k]]
        else:
            for k,v in val.items():
                v = to_item(v)
                self.sum[k] += v * n
                self.avg[k] = self.sum[k]/self.count
                self.display[k] = [v, self.avg[k]]

def levenshtein_distance(hypothesis: list, reference: list):
    '''WER (word error rate)'''
    len_hyp = len(hypothesis)
    len_ref = len(reference)
    cost_matrix = np.zeros((len_hyp + 1, len_ref + 1), dtype=np.int16)
    ops_matrix = np.zeros((len_hyp + 1, len_ref + 1), dtype=np.int8)

    for i in range(len_hyp + 1):
        cost_matrix[i][0] = i
    for j in range(len_ref + 1):
        cost_matrix[0][j] = j
    for i in range(1, len_hyp + 1):
        for j in range(1, len_ref + 1):
            if hypothesis[i-1] == reference[j-1]:
                cost_matrix[i][j] = cost_matrix[i-1][j-1]
            else:
                substitution = cost_matrix[i-1][j-1] + 1
                insertion = cost_matrix[i-1][j] + 1
                deletion = cost_matrix[i][j-1] + 1
                compare_val = [substitution, insertion, deletion]   # 优先级
                min_val = min(compare_val)
                operation_idx = compare_val.index(min_val) + 1
                cost_matrix[i][j] = min_val
                ops_matrix[i][j] = operation_idx

    match_idx = []
    i = len_hyp
    j = len_ref
    nb_map = {"N": len_ref, "C": 0, "W": 0, "I": 0, "D": 0, "S": 0}
    while i >= 0 or j >= 0:
        i_idx = max(0, i)
        j_idx = max(0, j)

        if ops_matrix[i_idx][j_idx] == 0:     # correct
            if i-1 >= 0 and j-1 >= 0:
                match_idx.append((j-1, i-1))
                nb_map['C'] += 1
            i -= 1
            j -= 1
        elif ops_matrix[i_idx][j_idx] == 2:   # insert
            i -= 1
            nb_map['I'] += 1
        elif ops_matrix[i_idx][j_idx] == 3:   # delete
            j -= 1
            nb_map['D'] += 1
        elif ops_matrix[i_idx][j_idx] == 1:   # substitute
            i -= 1
            j -= 1
            nb_map['S'] += 1
        if i < 0 and j >= 0:
            nb_map['D'] += 1
        elif j < 0 and i >= 0:
            nb_map['I'] += 1

    match_idx.reverse()
    wrong_cnt = cost_matrix[len_hyp][len_ref]
    nb_map["W"] = wrong_cnt
    return wrong_cnt, match_idx, nb_map

def is_Inf_or_Nan(loss):
    if isinstance(loss, list):
        loss = sum(loss)
    elif isinstance(loss, dict):
        loss = sum(list(loss.values()))
    if math.isinf(loss.item()) or math.isnan(loss.item()):
        print('batch index loss inf')
        return True
    return False



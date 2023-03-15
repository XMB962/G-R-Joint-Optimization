# 编解码工具
# imported by lib.operators.dataset

import torch

class strLabelConverter(object):
    '''解码工具'''
    def __init__(self, config, ignore_case=False):
        self.label2str = {}
        self.str2label = {}
        self.dict2num = {}
        self.num2dict = {}
        char_file = config.DATASET.CHAR_FILE
        dict_file = config.DATASET.DICT_FILE
        self.invalid_label = config.DATASET.VALID_LABEL

        with open(char_file, 'r', encoding='utf-8-sig') as file:
            for line in file.readlines():
                self.str2label[line[0]] = line[2:].strip('\n').strip(' ')
                self.label2str[line[2:].strip('\n').strip(' ')] = line[0]
        
        with open(dict_file, 'r', encoding='utf-8-sig') as file:
            for line in file.readlines():
                self.num2dict[int(line.split()[0])] = line.strip('\n').split()[1]
                self.dict2num[line.strip('\n').split()[1]] = int(line.split()[0])
        self.num2dict[config.DATASET.BLANK_LABEL] = ' '
        self.dict2num[' '] = config.DATASET.BLANK_LABEL

    def encode(self, text):

        if len(text)==1:
            item_label = self.str2label[text]
        else:
            item_label = text.replace('$', ' ')
        text = [self.dict2num[i] for i in item_label.split()]
        return torch.IntTensor(text)

    def decode(self, t, length, code=False, raw=False):
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(
                t.numel(), length)
            if raw:
                return [int(t[i]) for i in range(length)]
            else:
                char_list = []
                if code:
                    for i in range(length):
                        if t[i] != self.invalid_label and (not (i > 0 and t[i - 1] == t[i])):
                            char_list.append(self.alphabet[t[i] - 1])
                    return ''.join(char_list)
                else:
                    for i in range(length):
                        if t[i] != self.invalid_label and (not (i > 0 and t[i - 1] == t[i])):
                            char_list.append(t[i].item())
                    return char_list
        else:
            # batch mode
            assert t.numel() == length.sum(
            ), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts

    def decode_single(self, t, code=False, raw=False):
        lens = len(t)
        if raw:
            if code:
                return ''.join([self.alphabet[t[i] - 1] for i in range(lens)])
            else:
                return [int(t[i]) for i in range(lens)]
        else:
            char_list = []
            if code:
                for i in range(lens):
                    if t[i] != self.invalid_label and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
            else:
                for i in range(lens):
                    if t[i] != self.invalid_label and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(t[i])
                return char_list


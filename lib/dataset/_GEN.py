from __future__ import print_function, absolute_import
import torch.utils.data as data
import torch
import os
from caffe import DataLoader
import numpy as np
import cv2
import json
import random
import math

#---------------for Inpaint-------------------
class _GEN(data.Dataset):
    def __init__(self, config, converter, is_train=True):
        self.model_name = config.MODEL.NAME
        self.root = config.DATASET.ROOT
        self.max_label_len = config.DATASET.MAX_LEN
        self.converter = converter
        self.mean = np.array(config.DATASET.MEAN, dtype=np.float32)
        self.std = np.array(config.DATASET.STD, dtype=np.float32)
        self.eos = config.DATASET.END_LABEL
        self.valid = config.DATASET.VALID_LABEL
        self.crop_path = config.DATASET.CROP_PATH
        self.loader = DataLoader()
        if config.MODEL.NAME in ['generator']:
            self.POS_EMBED_LEN = config.MODEL.EN_POS_EMBED_LEN
        elif config.MODEL.NAME in ['recognizer']:
            self.POS_EMBED_LEN = config.MODEL.DE_POS_EMBED_LEN
        elif config.MODEL.NAME in ['joint']:
            assert config.MODEL.R.DE_POS_EMBED_LEN == config.MODEL.G.EN_POS_EMBED_LEN, 'error config'
            self.POS_EMBED_LEN = config.MODEL.R.DE_POS_EMBED_LEN
        
        txt_file = config.DATASET.JSON_FILE['train'] if is_train else config.DATASET.JSON_FILE['val']
        self.is_train = is_train
        self.loader.load(txt_file, True)
        if config.LOCAL_RANK == 0:
            print("\rank {} | load {} images!          ".format(config.LOCAL_RANK, self.__len__()))

    def __len__(self):
        return self.loader.count()

    def tree_node(self, label):
        # 0~9
        node = 0
        his = []
        result = []
        for c in label:
            if c == self.valid:
                break
            result.append(node)
            if 79<=c<=88: # spatial character
                his.append(node)
                node = node * 2 + 1
            else:
                while node != 0:
                    if node%2==0:
                        node = his[-1]
                        del(his[-1])
                    else:
                        node += 1
                        break
        l = len(result)
        result = np.array(result, dtype=np.float32)
        return result

    def __getitem__(self, idx):
        package = {}
        # init
        valid_len = 0
        mode = cv2.IMREAD_COLOR # IMREAD_GRAYSCALE
        success, image, labels, uri = self.loader.read(idx)
        # load data
        bug_idx = 0
        while not success:
            success, image, labels, uri = self.loader.read(bug_idx)
            bug_idx += 1
        # deal label
        gt_text = labels.decode('utf-8').strip('\n')
        gt_text = gt_text.replace('CODE_', 'CODE#')
        attribute = np.array([int(i) for i in gt_text.split('_')[1:4]], dtype=np.float32)
        is_syn, is_real = gt_text.split('_')[4:6]
        is_real = int(is_real)
        is_syn = int(is_syn)

        back_fn, ax, ay = 'white.png', 0, 0
        if len(gt_text.split('_'))>=9:
            back_fn, ax, ay = gt_text.split('_')[6:9]
            ax = float(ax)
            ay = float(ay)
        
        R,G,B = 0,0,0
        if len(gt_text.split('_'))>=12:
            R,G,B = [int(c) for c in gt_text.split('_')[9:]]

        back_fn = cv2.imread(os.path.join(self.crop_path, back_fn)).astype(np.float32)
        package['back_fn'] = back_fn.transpose(2,0,1)
        axy = [
            ax, math.tan(ax), math.cos(ax), 
            ay, math.tan(ay), math.cos(ay),
        ]
        package['axy'] = np.array(axy)
        package['rgb'] = np.array([R,G,B])

        gt_text = gt_text.split('_')[0].replace('CODE#', 'CODE_').replace('乂', '㐅')
        gt_text = self.converter.encode(gt_text)
        valid_len = len(gt_text)
        # deal image
        image = np.frombuffer(image, dtype=np.uint8)
        image = cv2.imdecode(image, mode)
        image = np.array(image, dtype=np.float32)
        # imgh, imgw = image.shape[0:2]

        # image = 255 * (image - np.min(image))/(np.max(image)-np.min(image))
        image = (image - self.mean) * self.std
        image = image.transpose(2,0,1)
        # image = np.reshape(image, (1, imgh, imgw))
        # data type
        labels = np.array([int(x) for x in gt_text], dtype=np.float32)
        tree_pos = self.tree_node(labels)
        labels = np.pad(labels, (0, self.max_label_len - labels.shape[0]), 'constant', constant_values=self.valid)
        tree_pos = np.pad(tree_pos, (0, self.max_label_len - tree_pos.shape[0]), 'constant', constant_values=self.POS_EMBED_LEN-1)
        # package
        package['image'] = image
        package['labels'] = labels
        package['valid_len'] = valid_len
        package['tree_pos'] = tree_pos
        package['attribute'] = attribute
        package['is_real'] = np.ones(1) * is_real
        package['is_syn'] = np.ones(1) * is_syn
        return package

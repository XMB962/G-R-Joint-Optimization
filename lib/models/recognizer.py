import torch
import torch.nn as nn

from lib.modules.self_attention import *
from lib.modules.mask import *
from lib.modules.embed import *
from lib.modules.dense_Conv import DenseConv_Maker

class VGG_FeatureExtractor(nn.Module):
    '''
        input channel : 1 (for gray image) or 3 (for RGB image)
        output channel : default 512
    '''
    def __init__(self, input_channel, output_channel=512):
        super(VGG_FeatureExtractor, self).__init__()
        self.output_channel = [int(output_channel // 8), int(output_channel // 4),
                               int(output_channel // 2), output_channel]  # [64, 128, 256, 512]
        self.ConvNet = nn.Sequential(
            nn.Conv2d(input_channel, self.output_channel[0], 3, 1, 1), 
            nn.BatchNorm2d(self.output_channel[0]), 
            nn.ReLU(True),
            nn.MaxPool2d(2, 2), 

            nn.Conv2d(self.output_channel[0], self.output_channel[1], 3, 1, 1), 
            nn.BatchNorm2d(self.output_channel[1]), 
            nn.ReLU(True),
            
            nn.Conv2d(self.output_channel[1], self.output_channel[2], 3, 1, 1), 
            nn.BatchNorm2d(self.output_channel[2]), 
            nn.ReLU(True),
            
            nn.Conv2d(self.output_channel[2], self.output_channel[2], 3, 1, 1), 
            nn.BatchNorm2d(self.output_channel[2]), 
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(self.output_channel[2], self.output_channel[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.output_channel[3]),
            nn.ReLU(True),

            nn.Conv2d(self.output_channel[3], self.output_channel[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.output_channel[3]),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2), 

            nn.Conv2d(self.output_channel[3], self.output_channel[3], 3, 1, 1), nn.ReLU(True)
        )

    def forward(self, input):
        return self.ConvNet(input)

class R_EncoderLayer(nn.Module):
    '''
        num hidden : default 512
        head num : default 4
        dim_hidden : dim in self-attn; num hidden / head num; default 128
    '''
    def __init__(self, num_hidden=512, num_head=4, dim_hidden=128):
        super(R_EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(num_hidden, num_head, dim_hidden, dp)
        self.pos_ffn = PoswiseFeedForwardNet(num_hidden, dp)

    def forward(self, enc_inputs, enc_self_attn_mask):
        # self attention; K = Q = V = enc_inputs
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        # feed forword
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn

class R_Encoder(nn.Module):
    '''
        num hidden : default 512
        head num : default 4
        dim_hidden : dim in self-attn; num hidden / head num; default 128
        n_layers : the number of R_Encoder layers; default 1
        EN_POS_EMBED_LEN : R_Encoder position embedding length; the length of image feature; default 16 = 4*4; image feature shape = H * W
    '''
    def __init__(self, num_hidden=512, num_head=4, dim_hidden=128, n_layers=1, EN_POS_EMBED_LEN=16):
        super(R_Encoder, self).__init__()
        self.pos_emb_x = nn.Embedding.from_pretrained(
            get_sinusoid_sequence_table(EN_POS_EMBED_LEN, num_hidden, dim=2, xy=True), 
            freeze=True
        )
        self.pos_emb_y = nn.Embedding.from_pretrained(
            get_sinusoid_sequence_table(EN_POS_EMBED_LEN, num_hidden, dim=2, xy=False),
            freeze=True
        )
        self.layers = nn.ModuleList([
            R_EncoderLayer(num_hidden, num_head, dim_hidden) for _ in range(n_layers)
        ])

    def forward(self, inp):
        device = inp.device
        b,c,h,w = inp.shape
        # pos embed
        pos_x = torch.linspace(0, h-1, h).to(device)
        pos_emb_x = self.pos_emb_x(pos_x.long())
        # pos_emb_x = pos_emb_x.unsqueeze(0).expand(b, h, c)
        pos_emb_x = pos_emb_x.permute(1,0).unsqueeze(-1).expand(c, h, w)

        pos_y = torch.linspace(0, w-1, w).to(device)
        pos_emb_y = self.pos_emb_y(pos_y.long())
        # pos_emb_y = pos_emb_y.unsqueeze(0).expand(b, w, c)
        pos_emb_y = pos_emb_y.permute(1,0).unsqueeze(-2).expand(c, h, w)
        # word + pos embed
        enc_outputs = inp + pos_emb_x + pos_emb_y
        enc_outputs = enc_outputs.view(b,c,-1).permute(0,2,1)
        # define mask
        enc_self_attn_mask = torch.zeros(b, h*w, h*w).bool().to(device)
        # N layers
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

class R_DecoderLayer(nn.Module):
    '''
        num hidden : default 512
        head num : default 4
        dim_hidden : dim in self-attn; num hidden / head num; default 128
    '''
    def __init__(self, num_hidden=512, num_head=4, dim_hidden=128):
        super(R_DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(num_hidden, num_head, dim_hidden, dp)
        self.dec_enc_attn = MultiHeadAttention(num_hidden, num_head, dim_hidden, dp)
        self.pos_ffn = PoswiseFeedForwardNet(num_hidden, dp)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        # self attention; K = Q = V = dec_inputs
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        # attention; K = dec_output; Q = V = enc_outputs
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        # feed forward
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn

class R_Decoder(nn.Module):
    '''
        num hidden : default 512
        head num : default 4
        dim_hidden : dim in self-attn; num hidden / head num; default 128
        n_layers : the number of R_Decoder layers; default 1
        DE_POS_EMBED_LEN : R_Decoder position embedding length; the maximum number of tree nodes/semantics feature; the max deepth of tree = log2(length + 1)
    '''
    def __init__(self, nclass, num_hidden=512, num_head=4, dim_hidden=128, n_layers=1, DE_POS_EMBED_LEN = 1023, VALID_LABEL=572, MAX_LEN=63, TREE_POS=True):
        super(R_Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(nclass, num_hidden)
        depth = math.ceil(math.log2(DE_POS_EMBED_LEN + 1))
        self.pos_emb_tree = nn.Embedding.from_pretrained(
            get_sinusoid_tree_table(depth, num_hidden), 
            freeze=True
        )
        self.layers = nn.ModuleList([
            R_DecoderLayer(num_hidden, num_head, dim_hidden) for _ in range(n_layers)
        ])
        self.projection = nn.Linear(num_hidden, nclass, bias=False)
        self.valid = VALID_LABEL
        self.tree_pos = TREE_POS
        self.dropout = nn.Dropout(p=dp)

    def forward(self, dec_inputs, tree_pos, enc_outputs):
        device = dec_inputs.device
        # word embed
        word_emb = self.tgt_emb(dec_inputs.long())
        # word_emb = self.dropout(word_emb)
        # pos embed
        if self.tree_pos:
            pos_emb = self.pos_emb_tree(tree_pos.long())
        else:
            pos_emb = torch.zeros(word_emb.shape).to(device)
        # word + position
        dec_outputs = word_emb + pos_emb
        # define mask
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs, self.valid).to(device)  
        dec_self_attn_subsequent_mask = get_attn_subsequence_mask(dec_inputs).to(device)
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), torch.tensor(0).to(device))
        dec_enc_attn_mask = torch.zeros(dec_inputs.shape[0], dec_inputs.shape[1], enc_outputs.shape[1]).bool().to(device)
        # N layers
        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        dec_outputs = self.projection(dec_outputs)
        return dec_outputs, dec_self_attns, dec_enc_attns

import sys
class Radical_Analysis_Network(nn.Module):
    def __init__(self, config):
        super(Radical_Analysis_Network, self).__init__()
        init_R_dp(config.TRAIN.DROPOUT)
        num_hidden = config.MODEL.NUM_HIDDEN
        nclass = config.MODEL.NUM_CLASS
        num_head = config.MODEL.NUM_HEAD
        dim_hidden = num_hidden//num_head
        n_layers = config.MODEL.NUM_LAYER
        EN_POS_EMBED_LEN = config.MODEL.EN_POS_EMBED_LEN
        DE_POS_EMBED_LEN = config.MODEL.DE_POS_EMBED_LEN
        VALID_LABEL = config.DATASET.VALID_LABEL
        MAX_LEN = config.DATASET.MAX_LEN
        TREE_POS = config.MODEL.TREE_POS
        # self.Down = VGG_FeatureExtractor(1, num_hidden)
        self.Down = DenseConv_Maker(input_channel=1, output_channel=num_hidden, growth_rate=32, block_config=(6, 12, 24, 16), bn_size=4, theta=0.5)
        self.R_Encoder = R_Encoder(num_hidden, num_head, dim_hidden, n_layers, EN_POS_EMBED_LEN=EN_POS_EMBED_LEN)
        self.R_Decoder = R_Decoder(nclass, num_hidden, num_head, dim_hidden, n_layers, DE_POS_EMBED_LEN=DE_POS_EMBED_LEN, VALID_LABEL=VALID_LABEL, MAX_LEN=MAX_LEN, TREE_POS=TREE_POS)
        # predict param
        self.max_len = config.DATASET.MAX_LEN
        self.st = config.DATASET.START_LABEL
        self.eos = config.DATASET.END_LABEL
        self.valid = config.DATASET.VALID_LABEL
        self.DE_POS_EMBED_LEN = DE_POS_EMBED_LEN

    def forward(self, inp):
        image, label, length, tree_pos = inp
        device = image.device
        batch_size = image.shape[0]
        pad_idx = torch.linspace(0, batch_size-1, batch_size).long()
        label[pad_idx, length.long()] = self.eos
        # add <start item>
        label = torch.cat([
            torch.ones(batch_size, 1).to(device) * self.st,
            label[:, :-1].float()
        ], dim=1)
        tree_pos = torch.cat([
            torch.ones(batch_size, 1).to(device) * (self.DE_POS_EMBED_LEN-1),
            tree_pos[:, :-1].float()
        ], dim=1)       
        # R_Encoder
        feature = self.Down(image)
        semantic,_ = self.R_Encoder(feature)
        # R_Decoder
        prediction,_,_ = self.R_Decoder(label, tree_pos, semantic)
        prediction = prediction.view(-1, prediction.size(-1))
        return prediction

    def node_init(self, B):
        self.node = [0 for _ in range(B)]
        self.his = [[] for _ in range(B)]

    def next_node(self, pred_s):
        for idx,c in enumerate(pred_s):
            if self.his[idx] is None:
                node = self.DE_POS_EMBED_LEN -1
            else:
                node = self.node[idx]
                if c == self.eos:
                    self.his[idx] = None
                    node = self.DE_POS_EMBED_LEN -1
                    self.node[idx] = node
                    continue
                if 79 <= c <= 88: # spatial character
                    self.his[idx].append(node)
                    node = node * 2 + 1
                else:
                    while node != 0:
                        if node%2==0:
                            node = self.his[idx][-1]
                            del(self.his[idx][-1])
                        else:
                            node += 1
                            break
                if node == 0:
                    assert self.his[idx] == [], 'len history[{}] = {}'.format(idx, len(self.his[idx]))
                    self.his[idx] = None
                    node = self.DE_POS_EMBED_LEN -1
            if node > self.DE_POS_EMBED_LEN - 1:
                self.his[idx] = None
                node = self.DE_POS_EMBED_LEN -1
            self.node[idx] = node
        return torch.tensor(self.node).to(pred_s.device)

    def pred(self, inp):
        image = inp
        device = image.device
        batch_size = image.shape[0]
        # R_Encoder
        feature = self.Down(image)
        semantic, enc_self_attns = self.R_Encoder(feature)
        self.enc_attn = enc_self_attns
        # R_Decoder
        self.node_init(batch_size)
        result = torch.zeros(batch_size, self.max_len).to(device)
        s = torch.ones(batch_size, 1).to(device) * self.st
        tree_pos = torch.ones(batch_size, 1).to(device) * (self.DE_POS_EMBED_LEN - 1)
        continue_flg = torch.ones(batch_size).long().to(device)
        for i in range(self.max_len):
            prediction, _, dec_enc_attns = self.R_Decoder(s, tree_pos, semantic)
            _, s_r = torch.max(prediction, dim=-1)
            s_r = s_r[:, -1].view(-1)
            continue_flg *= (s_r != self.eos)
            s = torch.cat([s, s_r.float().view(batch_size, 1)], dim=-1)
            if i == 0:
                pos_0 = torch.zeros(batch_size, 1).to(device)
                tree_pos = torch.cat([tree_pos, pos_0], dim=-1)
            else:
                tree_pos = torch.cat([tree_pos, self.next_node(s_r_his).float().view(batch_size, 1)], dim=-1)
            s_r_his = s_r

            if torch.sum(continue_flg) == 0:
                self.dec_attn = dec_enc_attns
                break
        result[:, :i+1] = s[:, 1:].view(batch_size, -1)
        return result

def weights_init(m):
    classname = m.__class__.__name__
    try:
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
    except:
        pass


def get_model(config):
    model = Radical_Analysis_Network(config)
    model.apply(weights_init)
    return model

def init_R_dp(param):
    global dp
    dp = param
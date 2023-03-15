import random

import torch
import torch.nn as nn

from lib.modules.self_attention import *
from lib.modules.mask import *
from lib.modules.embed import *
from lib.modules.dense_Deconv import DenseDeconv_Maker


class G_EncoderLayer(nn.Module):
    '''
        num hidden : default 512
        head num : default 4
        dim_hidden : dim in self-attn; num hidden / head num; default 128
    '''
    def __init__(self, num_hidden=512, head_num=4, dim_hidden=128):
        super(G_EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(num_hidden, head_num, dim_hidden, dp)
        self.pos_ffn = PoswiseFeedForwardNet(num_hidden, dp)

    def forward(self, enc_inputs, enc_self_attn_mask):
        # self attention; K = Q = V = enc_inputs
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        # feed forword
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn

class G_Encoder(nn.Module):
    '''
        num hidden : default 512
        head num : default 4
        dim_hidden : dim in self-attn; num hidden / head num; default 128
        n_layers : the number of G_Encoder layers; default 1
        EN_POS_EMBED_LEN : G_Encoder position embedding length; the maximum number of tree nodes/semantics feature; the max deepth of tree = log2(length + 1)
    '''
    def __init__(self, nclass, num_hidden=512, head_num=4, dim_hidden=128, n_layers=1, EN_POS_EMBED_LEN=1023, VALID_LABEL=515, MAX_LEN=63, TREE_POS=True):
        super(G_Encoder, self).__init__()
        self.tgt_emb = nn.Embedding(nclass, num_hidden)
        depth = math.ceil(math.log2(EN_POS_EMBED_LEN + 1))
        self.pos_emb_tree = nn.Embedding.from_pretrained(
            get_sinusoid_tree_table(depth, num_hidden), 
            freeze=True
        )
        self.layers = nn.ModuleList([
            G_EncoderLayer(num_hidden, head_num, dim_hidden) for _ in range(n_layers)
        ])
        self.EN_POS_EMBED_LEN = EN_POS_EMBED_LEN
        self.MAX_LEN = MAX_LEN
        self.num_hidden = num_hidden
        self.valid = VALID_LABEL
        self.tree_pos = TREE_POS
        self.dropout = nn.Dropout(p=dp)

    def forward(self, enc_inp, tree_pos):
        device = enc_inp.device
        # word embedding
        word_emb = self.tgt_emb(enc_inp.long())
        word_emb = self.dropout(word_emb)
        # position embedding
        if self.tree_pos:
            pos_emb = self.pos_emb_tree(tree_pos.long())
        else:
            pos_emb = torch.zeros(word_emb.shape).to(device)
        # word + position
        enc_outputs = word_emb + pos_emb
        # mask; tree and padding
        enc_self_attn_mask = get_attn_pad_mask(enc_inp, enc_inp, self.valid).to(device)
        # N layers
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

class G_DecoderLayer(nn.Module):
    '''
        num hidden : default 512
        head num : default 4
        dim_hidden : dim in self-attn; num hidden / head num; default 128
    '''
    def __init__(self, num_hidden=512, head_num=4, dim_hidden=128):
        super(G_DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(num_hidden, head_num, dim_hidden, dp)
        self.dec_enc_attn = MultiHeadAttention(num_hidden, head_num, dim_hidden, dp)
        self.pos_ffn = PoswiseFeedForwardNet(num_hidden, dp)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        # self attention; K = Q = V = dec_inputs
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        # attention; K = dec_output; Q = V = enc_outputs
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        # feed forward
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn

class G_Decoder(nn.Module):
    '''
        num hidden : default 512
        head num : default 4
        dim_hidden : dim in self-attn; num hidden / head num; default 128
        n_layers : the number of G_Decoder layers; default 1
        DE_POS_EMBED_LEN : G_Decoder position embedding length; the length of image feature; default 16 = 4*4; image feature shape = H * W
    '''
    def __init__(self, num_hidden=512, head_num=4, dim_hidden=128, n_layers=1, DE_POS_EMBED_LEN = 16, VALID_LABEL=515):
        super(G_Decoder, self).__init__()
        self.writer_emb = nn.Embedding(250, num_hidden*16)
        from lib.models.recognizer import VGG_FeatureExtractor
        self.feature_extract = VGG_FeatureExtractor(input_channel=3)
        self.axy_linear = nn.Sequential(
            nn.Linear(6, num_hidden),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(num_hidden, num_hidden*16),
        )
        self.rgb_emb = nn.Embedding(17, num_hidden*16)
        self.random_seed = nn.Embedding(250, num_hidden*16)
        self.dec_weight = torch.nn.Parameter(torch.FloatTensor(4))
        self.rgb_weight = torch.nn.Parameter(torch.FloatTensor(3))
        self.dropout = nn.Dropout(p=dp)
        DE_EMBED_XY = int(math.sqrt(DE_POS_EMBED_LEN))
        self.pos_emb_x = nn.Embedding.from_pretrained(
            get_sinusoid_sequence_table(DE_EMBED_XY, num_hidden, dim=2, xy=True), 
            freeze=True
        )
        self.pos_emb_y = nn.Embedding.from_pretrained(
            get_sinusoid_sequence_table(DE_EMBED_XY, num_hidden, dim=2, xy=False), 
            freeze=True
        )
        self.H = self.W = DE_EMBED_XY
        self.layers = nn.ModuleList([
            G_DecoderLayer(num_hidden, head_num, dim_hidden) for _ in range(n_layers)
        ])
        self.valid = VALID_LABEL


    def forward(self, dec_inputs, enc_inp, enc_outputs, back_fn=None, axy=None, rgb=None):
        device = enc_outputs.device
        B,_,C = enc_outputs.shape
        # feature embedding
        dec_writer = self.writer_emb(dec_inputs[:,0].long())
        if back_fn is not None:
            dec_back = self.feature_extract(back_fn).view(B,-1)
        if axy is not None:
            dec_axy = self.axy_linear(axy.float())
        dec_rgb = self.rgb_emb(rgb.long())
        dec_random_seed = self.random_seed(torch.randint(0,250,(B,1)).to(device).squeeze().long())
        dec_output = dec_writer + self.dec_weight[0] * dec_back + \
            self.dec_weight[1] * dec_axy + self.dec_weight[2] * dec_random_seed + \
            self.dec_weight[3] * (dec_rgb * self.rgb_weight.view(1,-1,1)).sum(1)
        dec_output = self.dropout(dec_output)
        dec_outputs = dec_output.view(B,16,-1)
        B,T,C = dec_outputs.shape
        # pos embed
        pos_x = torch.linspace(0, self.H-1, self.H).to(device)
        pos_emb_x = self.pos_emb_x(pos_x.long())
        pos_emb_x = pos_emb_x.unsqueeze(0).expand(B,self.H,C)
        pos_emb_x = pos_emb_x.permute(0,2,1).unsqueeze(-1).expand(B,C,self.H,self.W)
        pos_emb_x = pos_emb_x.contiguous().view(B,C,-1).permute(0,2,1)

        pos_y = torch.linspace(0, self.W-1, self.W).to(device)
        pos_emb_y = self.pos_emb_y(pos_y.long())
        pos_emb_y = pos_emb_y.unsqueeze(0).expand(B,self.W,C)
        pos_emb_y = pos_emb_y.permute(0,2,1).unsqueeze(-2).expand(B,C,self.H,self.W)
        pos_emb_y = pos_emb_y.contiguous().view(B,C,-1).permute(0,2,1)
        # word + pos embed
        dec_outputs = dec_outputs + pos_emb_x + pos_emb_y
        # define mask
        dec_self_attn_mask = torch.zeros(B, T, T).bool().to(device)
        dec_enc_attn_mask = get_attn_pad_mask(dec_outputs, enc_inp, self.valid, blank_pad=True).to(device)
        # N layer
        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        dec_outputs = dec_outputs.view(B,self.H,self.W,C).permute(0,3,1,2)
        return dec_outputs, dec_self_attns, dec_enc_attns

class Trans_conv_2d(nn.Module):
    '''
        input channel : default 512
        output channel : 1 (for gray image) or 3 (for RGB image)
    '''
    def __init__(self, input_channel, output_channel):
        super(Trans_conv_2d, self).__init__()
        output_channel = [
            int(input_channel // 2), 
            int(input_channel // 4),
            int(input_channel // 8), 
            output_channel
        ]  # [64, 128, 256, 512]
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels=input_channel, out_channels=output_channel[0], kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(output_channel[0], track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=False),

            nn.ConvTranspose2d(in_channels=output_channel[0], out_channels=output_channel[1], kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(output_channel[1], track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=False),

            nn.ConvTranspose2d(in_channels=output_channel[1], out_channels=output_channel[1], kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(output_channel[1], track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=False),

            nn.ConvTranspose2d(in_channels=output_channel[1], out_channels=output_channel[2], kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(output_channel[2], track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=False),

            nn.ConvTranspose2d(in_channels=output_channel[2], out_channels=output_channel[2], kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(output_channel[2], track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=False),

            nn.ConvTranspose2d(in_channels=output_channel[2], out_channels=output_channel[3], kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
    def forward(self, inp):
        return self.model(inp)

class Radical_Combine_Network(nn.Module):
    def __init__(self, config):
        super(Radical_Combine_Network, self).__init__()
        init_G_dp(config.TRAIN.DROPOUT)
        num_hidden = config.MODEL.NUM_HIDDEN
        nclass = config.MODEL.NUM_CLASS
        head_num = config.MODEL.NUM_HEAD
        dim_hidden = num_hidden//head_num
        n_layers = config.MODEL.NUM_LAYER
        EN_POS_EMBED_LEN = config.MODEL.EN_POS_EMBED_LEN
        DE_POS_EMBED_LEN = config.MODEL.DE_POS_EMBED_LEN
        VALID_LABEL = config.DATASET.VALID_LABEL
        MAX_LEN = config.DATASET.MAX_LEN
        TREE_POS = config.MODEL.TREE_POS
        self.G_Encoder = G_Encoder(nclass, num_hidden, head_num, dim_hidden, n_layers, EN_POS_EMBED_LEN=EN_POS_EMBED_LEN, VALID_LABEL=VALID_LABEL, MAX_LEN=MAX_LEN, TREE_POS=TREE_POS)
        self.G_Decoder = G_Decoder(num_hidden, head_num, dim_hidden, n_layers, DE_POS_EMBED_LEN=DE_POS_EMBED_LEN, VALID_LABEL=VALID_LABEL)
        self.Up = Trans_conv_2d(num_hidden, 1)
        # self.Up = DenseDeconv_Maker(input_channel=num_hidden, output_channel=1, growth_rate=32, block_config=(6, 12, 24, 16), bn_size=4, theta=0.5)

    def forward(self, inp):
        label, tree_pos, attribute = inp
        semantic, enc_self_attn = self.G_Encoder(label, tree_pos)
        feature,_,dec_enc_attns = self.G_Decoder(attribute, label, semantic)
        self.enc_attn = enc_self_attn
        self.dec_attn = dec_enc_attns
        fake_image = self.Up(feature)
        return fake_image, dec_enc_attns[-1]

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_model(config):
    model = Radical_Combine_Network(config)
    model.apply(weights_init)
    return model

def init_G_dp(param):
    global dp
    dp = param
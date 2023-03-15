import torch
import torch.nn as nn

from lib.models.generator import G_Encoder, G_Decoder, Trans_conv_2d, init_G_dp

from lib.models.recognizer import VGG_FeatureExtractor, R_Encoder, R_Decoder, init_R_dp

from lib.modules.dense_Deconv import DenseDeconv_Maker

from lib.modules.dense_Conv import DenseConv_Maker

from lib.utils.gradient import compute_gradient_penalty

from lib.modules.contrastive import contrastive_mapping

class Joint_Learning_Network(nn.Module):
    def __init__(self, config):
        super(Joint_Learning_Network, self).__init__()
        init_R_dp(config.TRAIN.DROPOUT)
        init_G_dp(config.TRAIN.DROPOUT)
        num_hidden = config.MODEL.NUM_HIDDEN
        nclass = config.MODEL.NUM_CLASS
        head_num = config.MODEL.NUM_HEAD
        dim_hidden = num_hidden//head_num
        n_layers = config.MODEL.NUM_LAYER
        VALID_LABEL = config.DATASET.VALID_LABEL
        MAX_LEN = config.DATASET.MAX_LEN
        # Recognizer
        R_EN_POS_EMBED_LEN = config.MODEL.R.EN_POS_EMBED_LEN
        R_DE_POS_EMBED_LEN = config.MODEL.R.DE_POS_EMBED_LEN
        self.R_Down = VGG_FeatureExtractor(3, num_hidden) # DenseConv_Maker(input_channel=1, output_channel=num_hidden, growth_rate=32, block_config=(6, 12, 24, 16), bn_size=4, theta=0.5) # 
        self.R_Encoder = R_Encoder(num_hidden, head_num, dim_hidden, n_layers, EN_POS_EMBED_LEN=R_EN_POS_EMBED_LEN)
        self.R_Decoder = R_Decoder(nclass, num_hidden, head_num, dim_hidden, n_layers, DE_POS_EMBED_LEN=R_DE_POS_EMBED_LEN, VALID_LABEL=VALID_LABEL, MAX_LEN=MAX_LEN)
        # Generator
        G_EN_POS_EMBED_LEN = config.MODEL.G.EN_POS_EMBED_LEN
        G_DE_POS_EMBED_LEN = config.MODEL.G.DE_POS_EMBED_LEN
        self.G_Encoder = G_Encoder(nclass, num_hidden, head_num, dim_hidden, n_layers, EN_POS_EMBED_LEN=G_EN_POS_EMBED_LEN, VALID_LABEL=VALID_LABEL, MAX_LEN=MAX_LEN)
        self.G_Decoder = G_Decoder(num_hidden, head_num, dim_hidden, n_layers, DE_POS_EMBED_LEN=G_DE_POS_EMBED_LEN, VALID_LABEL=VALID_LABEL)
        self.G_Up = Trans_conv_2d(num_hidden, 3) # DenseDeconv_Maker(input_channel=num_hidden, output_channel=1, growth_rate=32, block_config=(6, 12, 24, 16), bn_size=4, theta=0.5) # 
        # predict param
        self.max_len = config.DATASET.MAX_LEN
        self.st = config.DATASET.START_LABEL
        self.eos = config.DATASET.END_LABEL
        self.valid = config.DATASET.VALID_LABEL
        self.R_DE_POS_EMBED_LEN = R_DE_POS_EMBED_LEN
        # pattern select
        self.if_Dual = config.TRAIN.PATTERN.IF_DUAL
        self.if_SIM = config.TRAIN.PATTERN.IF_SIM
        self.is_gan_str = config.TRAIN.PATTERN.GAN
        if self.is_gan_str:
            from lib.modules.discriminator import Discriminator
            self.Discriminator_str = Discriminator(input_channel=3)
        self.is_gan_syn = config.TRAIN.PATTERN.SYN
        if self.is_gan_syn:
            from lib.modules.discriminator import Discriminator
            self.Discriminator_syn = Discriminator(input_channel=3)
        self.freeze_encoder = config.TRAIN.FREEZE_ENCODER
        self.contrast_mapping = contrastive_mapping(num_hidden, num_hidden)

    def forward(self, package):
        device = package['device']
        # unpack
        image = package['image'].to(device)
        label = package['labels'].to(device)
        length = package['valid_len'].to(device)
        tree_pos = package['tree_pos'].to(device)
        attribute = package['attribute'].to(device)
        rgb = package['rgb'].to(device)
        try:
            back_fn = package['back_fn'].to(device)
            axy = package['axy'].to(device)
        except:
            back_fn = None
            axy = None

        batch_size = image.shape[0]
        result = {}

        # Generator
        semantic_g, _ = self.G_Encoder(label, tree_pos)
        if self.freeze_encoder:
            semantic_g = semantic_g.detach()
        feature_g, _, G_dec_enc_attns = self.G_Decoder(attribute, label, semantic_g, back_fn, axy, rgb)
        fake_image = self.G_Up(feature_g)
        result['fake_image'] = fake_image
        result['G_attns'] = G_dec_enc_attns

        # Recognizer
        pad_idx = torch.linspace(0, batch_size-1, batch_size).long()
        label[pad_idx, length.long()] = self.eos
        label = torch.cat([
            torch.ones(batch_size, 1).to(device) * self.st,
            label[:, :-1].float()
        ], dim=1)
        tree_pos = torch.cat([
            torch.ones(batch_size, 1).to(device) * (self.R_DE_POS_EMBED_LEN-1),
            tree_pos[:, :-1].float()
        ], dim=1)
        feature_r = self.R_Down(image)
        semantic, _ = self.R_Encoder(feature_r)
        prediction_r, _, R_dec_enc_attns = self.R_Decoder(label, tree_pos, semantic)
        prediction = prediction_r.view(-1, prediction_r.size(-1))
        result['prediction'] = prediction
        result['R_attns'] = R_dec_enc_attns

        if self.if_Dual:
            feature = self.R_Down(fake_image)
            semantic, _ = self.R_Encoder(feature)
            fake_prediction, _, fake_R_dec_enc_attns = self.R_Decoder(label, tree_pos, semantic)
            fake_prediction = fake_prediction.view(-1, fake_prediction.size(-1))
            result['DUAL_prediction'] = fake_prediction
            result['DUAL_R_attns'] = fake_R_dec_enc_attns

            semantic, _ = self.G_Encoder(prediction, tree_pos)
            if self.freeze_encoder:
                semantic = semantic.detach()
            feature, _, G_dec_enc_attns = self.G_Decoder(attribute, label, semantic, back_fn, axy, rgb)
            fake_image = self.G_Up(feature)
            result['DUAL_fake_image'] = fake_image
            result['DUAL_G_attns'] = G_dec_enc_attns
        
        if self.if_SIM:
            # 未完全整理，此处有bug
            X_G = self.contrast_mapping(semantic_g)
            X_R = self.contrast_mapping(prediction_r)
            Y_G = self.contrast_mapping(feature_g)
            Y_R = self.contrast_mapping(feature_r)
            result['sim_x'] = torch.matmul(X_G, X_R.permute(0,2,1))
            result['sim_y'] = torch.matmul(Y_G, Y_R.permute(0,2,1))

        if self.is_gan_str:
            result['D_real_str'] = self.Discriminator_str(image)
            result['D_fake_str'] = self.Discriminator_str(fake_image)
            result['D_penalty_str'] = compute_gradient_penalty(self.Discriminator_str, fake_image)
        
        if self.is_gan_syn:
            result['D_real_syn'] = self.Discriminator_syn(image)
            result['D_fake_syn'] = self.Discriminator_syn(fake_image)
            result['D_penalty_syn'] = compute_gradient_penalty(self.Discriminator_syn, fake_image)

        return result

    def node_init(self, B):
        self.node = [0 for _ in range(B)]
        self.his = [[] for _ in range(B)]

    def next_node(self, pred_s):
        for idx,c in enumerate(pred_s):
            if self.his[idx] is None:
                node = self.R_DE_POS_EMBED_LEN -1
            else:
                node = self.node[idx]
                if c == self.eos:
                    self.his[idx] = None
                    node = self.R_DE_POS_EMBED_LEN -1
                    self.node[idx] = node
                    continue
                if 79<=c<=88: # spatial character
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
                    node = self.R_DE_POS_EMBED_LEN -1
            if node > self.R_DE_POS_EMBED_LEN - 1:
                self.his[idx] = None
                node = self.R_DE_POS_EMBED_LEN -1
            self.node[idx] = node
        return torch.tensor(self.node).to(pred_s.device)

    def pred(self, package):
        device = package['device']
        # unpack
        image = package['image'].to(device)
        label = package['labels'].to(device)
        length = package['valid_len'].to(device)
        tree_pos = package['tree_pos'].to(device)
        attribute = package['attribute'].to(device)
        rgb = package['rgb'].to(device)
        try:
            back_fn = package['back_fn'].to(device)
            axy = package['axy'].to(device)
        except:
            back_fn = None
            axy = None
        result = {}
        batch_size = image.shape[0]

        # Generator
        semantic, _ = self.G_Encoder(label, tree_pos)
        feature, _, G_dec_enc_attns = self.G_Decoder(attribute, label, semantic, back_fn, axy, rgb)
        self.G_attn = G_dec_enc_attns
        fake_image = self.G_Up(feature)
        result['fake_image'] = fake_image
        print('fake loss',nn.MSELoss()(fake_image, image))

        # Recognizer
        feature = self.R_Down(image)
        semantic, enc_self_attns = self.R_Encoder(feature)
        self.enc_attn = enc_self_attns
        # R_Decoder
        self.node_init(batch_size)
        preds = torch.zeros(batch_size, self.max_len).to(device)
        s = torch.ones(batch_size, 1).to(device) * self.st
        tree_pos = torch.ones(batch_size, 1).to(device) * (self.R_DE_POS_EMBED_LEN - 1)
        continue_flg = torch.ones(batch_size).long().to(device)
        for i in range(self.max_len):
            prediction, _, dec_enc_attns = self.R_Decoder(s, tree_pos, semantic)
            self.R_attn = dec_enc_attns
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
                self.R_attn = dec_enc_attns
                break
        preds[:, :i+1] = s[:, 1:].view(batch_size, -1)
        result['predict'] = preds

        return result

def weights_init(m):
    classname = m.__class__.__name__
    if 'Conv' in classname and 'Maker' not in classname:
        m.weight.data.normal_(0.0, 0.02)
    elif 'BatchNorm' in classname:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_model(config):
    model = Joint_Learning_Network(config)
    model.apply(weights_init)
    return model




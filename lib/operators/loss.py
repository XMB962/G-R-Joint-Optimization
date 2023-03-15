import torch
import torch.nn as nn
import random
class joint_loss(nn.Module):
    def __init__(self, config, ignore=515):
        super(joint_loss, self).__init__()
        self.CE = torch.nn.CrossEntropyLoss(ignore_index=ignore, reduction = 'none')
        self.CE_mean = torch.nn.CrossEntropyLoss(ignore_index=ignore, reduction = 'mean')
        self.MSE = torch.nn.MSELoss(reduction = 'none')
        self.MSE_attn = torch.nn.MSELoss(reduction = 'none')
        self.is_joint = config.TRAIN.PATTERN.JOINT
        self.if_Dual = config.TRAIN.PATTERN.IF_DUAL
        self.if_SIM = config.TRAIN.PATTERN.IF_SIM
        self.is_gan = config.TRAIN.PATTERN.GAN
        self.is_syn = config.TRAIN.PATTERN.SYN

    def forward(self, preds, image, label, length, Real_or_Fake, Syn_or_Structure):
        batchsize = image.shape[0]
        device = image.device
        loss = {}
        RF_bool = (Real_or_Fake==1.).float()
        if random.randint(0,200) == 66: #random turn label
            SYN_bool = (Syn_or_Structure!=0.).float()
        else:
            SYN_bool = (Syn_or_Structure==0.).float()
        # get results
        fake_image = preds['fake_image']
        G_dec_enc_attns = preds['G_attns']
        prediction = preds['prediction']
        R_dec_enc_attns = preds['R_attns']
        # define mask and target
        mask = torch.zeros(G_dec_enc_attns[0].shape).to(device)
        ZERO = torch.zeros(G_dec_enc_attns[0].shape).to(device)
        for i in range(batchsize):
            mask[i,:,:,:length[i]] = 1.
        # recognizer loss
        loss_R = self.CE(prediction, label.long().view(-1)).view(batchsize, -1).sum(-1)/length
        loss['R'] = torch.sum(loss_R * RF_bool)/torch.sum(RF_bool + 1e-9)
        # image loss and entropy loss
        loss_G = self.MSE(fake_image, image).mean(-1).mean(-1).mean(-1)
        loss['G'] = torch.sum(loss_G * RF_bool)/torch.sum(RF_bool +  + 1e-9)
        # similarity loss
        if self.is_joint:
            r_attn = R_dec_enc_attns[-1].permute(0,1,3,2)
            g_attn = G_dec_enc_attns[-1]
            loss_J = self.MSE_attn(g_attn-r_attn, ZERO) * mask
            loss_J = loss_J.sum(-1).sum(-1).sum(-1) / mask.sum(-1).sum(-1).sum(-1)
            loss['J'] = torch.sum(loss_J * RF_bool)/torch.sum(RF_bool + 1e-9)
        # recognize fake image
        if self.if_Dual:
            fake_prediction = preds['DUAL_prediction']
            fake_fake_image = preds['DUAL_fake_image']

            loss_R_F = self.CE_mean(fake_prediction, label.long().view(-1))
            loss_R_G = self.MSE(fake_fake_image, image).mean(-1).mean(-1).mean(-1)

            loss['R_F'] = loss_R_F
            loss['R_G'] = loss_R_G

        if self.if_SIM:
            sim_x = torch.exp(preds['sim_x'])
            sim_y = torch.exo(preds['sim_y'])
            loss_sim_x = torch.sum(torch.diag(sim_x)) / torch.sum(sim_x * torch.triu(sim_x, k=0))
            loss_sim_y = torch.sum(torch.diag(sim_y)) / torch.sum(sim_y * torch.triu(sim_y, k=0))
            loss['S_X'] = loss_sim_x
            loss['S_Y'] = loss_sim_y


        # GAN
        if self.is_gan:
            fake_str_test_loss = torch.sum(preds['D_fake_str'].squeeze() * (1 - RF_bool))
            fake_str_train_loss = torch.sum(preds['D_fake_str'].squeeze() * RF_bool)
            loss['D_R_str'] = (preds['D_real_str'].squeeze().sum(-1) + fake_str_train_loss) / (batchsize + torch.sum(RF_bool))
            loss['D_F_str'] = fake_str_test_loss / torch.sum(1 - RF_bool + 1e-9)
            loss['D_P_str'] = preds['D_penalty_str']
        if self.is_syn:
            fake_syn_loss = torch.sum(preds['D_fake_syn'].squeeze() * (1 - SYN_bool))
            loss['D_R_syn'] = torch.sum(preds['D_real_syn'].squeeze() * (1 - SYN_bool)) / torch.sum(1 - SYN_bool + 1e-9)
            loss['D_F_syn'] = fake_syn_loss / torch.sum(1 - SYN_bool + 1e-9)
            loss['D_P_syn'] = preds['D_penalty_syn']

        return loss



def get_criterion(config):
    # 具有临时性
    def gen_loss_map():
        LOSS_DICT = {}
        LOSS_DICT["CE"] = torch.nn.CrossEntropyLoss(ignore_index=config.DATASET.VALID_LABEL, reduction='mean')
        LOSS_DICT["MSE"] = torch.nn.MSELoss(reduction='mean')
        LOSS_DICT["JOINT"] = joint_loss(config, ignore=config.DATASET.VALID_LABEL)
        return LOSS_DICT

    return gen_loss_map()[config.MODEL.CRITERION]

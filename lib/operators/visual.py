import torch
import numpy as np
import pygame
import sys
from PIL import Image
import cv2
import time
from lib.utils.testing import decode_pred
pygame.init()

def make_a_char(font, c, color, back):
    try:
        ftext = font.render(c, True, color, back)
    except:
        ftext = font.render('卍', True, color, back)
    pygame.image.save(ftext, "p_char.jpg")
    try:
        p_im = Image.open('p_char.jpg')
    except:
        time.sleep(0.1)
        p_im = Image.open('p_char.jpg')
    return p_im

def tensor2image(inp):
    inp = inp.permute(0,2,3,1).squeeze()
    inp = inp.detach().cpu().numpy()
    return inp

def link_attn(attn):
    attn = [e.sum(1) for e in attn]
    attn_map = attn[-1]
    return attn_map.detach().cpu().numpy()

def num2char(line, converter):
    line = [converter.num2dict[i] for i in line]
    line = [i if len(i)==1 else '&' for i in line]
    line = ''.join(line).replace('$', '')
    line = list(line)
    return line

def attn_visual_img2str(inp, dec, pred, label, st, config, converter):
    attn_map = link_attn(dec)
    batchsize,L,T = attn_map.shape
    inp = inp.cpu().numpy()
    font = pygame.font.Font('lib/core/a.ttf', 25)

    for bi in range(batchsize):
        print('    Drawing R Attn {}'.format(bi+st*batchsize))
        im = inp[bi,0,:,:]
        im = 255*(im-np.min(im))/(np.max(im)-np.min(im))
        
        im_list = []

        p_line = decode_pred(config, pred[bi].int().detach().cpu().tolist(), converter)
        p_line = num2char(p_line, converter)
        t_line = num2char(label[bi], converter)

        for li in range(L):
            if li>=len(p_line) and li>=len(t_line):
                break
            img = im.copy()
            attn = attn_map[bi,li,:]
            attn = attn.reshape((4,4))
            attn = attn.repeat(8, axis=0).repeat(8, axis=1)
            attn = np.expand_dims(attn, axis=2).repeat(2, axis=2)
            attn *= 255
            attn_r = np.zeros((attn.shape[0],attn.shape[1],1))
            attn = np.concatenate([attn_r, attn], axis=-1)
            img = Image.fromarray(img).convert('RGB')
            img = np.array(img) * -1
            img = (img + attn) * -1
            
            for i in range(attn.shape[0]):
                for j in range(attn.shape[1]):
                    for k in range(3):
                        if img[i,j,k]>255:
                            img[i,j,k] = 255
                        if img[i,j,k]<0:
                            img[i,j,k] = 0
            img = Image.fromarray(np.uint8(img))

            p_char = p_line[li] if li<len(p_line) else ' '
            t_char = t_line[li] if li<len(t_line) else ' '
            color = (0,200,0) if p_char==t_char else (200,0,0)

            p_im = make_a_char(font, p_char, color, (255,255,255))
            t_im = make_a_char(font, t_char, (0,0,0), (255,255,255))
            im_list.append([img, p_im, t_im])

        back = Image.new('RGB', (32+54, 32*len(im_list)), (255,255,255))
        for idx,(img, p_im, t_im) in enumerate(im_list):
            back.paste(img, (0,idx*32))
            back.paste(p_im, (32+2,idx*32))
            back.paste(t_im, (32+2+25+2,idx*32))
        back.save(config.MODEL.RESULT_SAVE_PATH + '/visual_R_{}.jpg'.format(bi+st*batchsize))

def attn_map_img2str(inp, dec, pred, label, st, config, converter):
    font = pygame.font.Font('lib/core/a.ttf', 26)
    attn_map = link_attn(dec)
    B,L,T = attn_map.shape
    inp = tensor2image(inp)

    for bi in range(B):
        print('    Drawing R map {}'.format(bi+st*B))
        p_line = decode_pred(config, pred[bi].int().detach().cpu().tolist(), converter)
        p_line = num2char(p_line, converter) + ['E']
        lens = len(p_line)
        back = Image.new('RGB', (lens*32+35, 16*32+35), (255,255,255))

        block = np.ones((8,8,3),dtype=np.float32) * 50 
        block[:,:,1:] *= -1
        im = inp[bi]
        src = 255*(im-np.min(im))/(np.max(im)-np.min(im))
        
        src = np.expand_dims(src, axis=-1).repeat(3, axis=-1)
        for i in range(16):
            im = src.copy()
            x = i//4
            y = i%4
            im[x*8:(x+1)*8,y*8:(y+1)*8] += block
            im = np.clip(im,0,255)
            im = Image.fromarray(np.uint8(im))
            back.paste(im,(0,i*32))

        for i,c in enumerate(p_line):
            im = make_a_char(font, c, (0,0,0), (255,255,255))
            back.paste(im, (35+i*32+3,16*32+3))

        attn = attn_map[bi,:lens,:].transpose()
        attn = attn.repeat(32,axis=0).repeat(32,axis=1)
        attn = np.expand_dims(attn, -1)
        attn = 255 - attn.repeat(2,axis=2) * 255
        attn = np.clip(attn,0,255)
        red = np.ones((16*32,lens*32,1))*255
        attn = np.concatenate([red, attn], axis=-1)
        attn = np.uint8(attn)
        attn = Image.fromarray(attn)
        back.paste(attn, (35,0))
        back.save(config.MODEL.RESULT_SAVE_PATH + '/map_R_{}.jpg'.format(bi+st*B))

def attn_visual_str2img(inp, dec, pred, label, st, config, converter):
    font = pygame.font.Font('lib/core/a.ttf', 20)
    savepath = config.MODEL.RESULT_SAVE_PATH
    def make_line(line, score):
        line = line + ['B']
        L = len(line)
        new_score = np.zeros((L), dtype=np.float32)
        new_score[:-1] = score[:L-1]
        new_score[-1] = np.sum(score[L-1:])
        back = Image.new('RGB',(20*L,32),(255,255,255))
        for i,t in enumerate(line):
            BG = max(0, int(255*(1-new_score[i])))
            c_im = make_a_char(font, t, (0,0,0), (255, BG, BG))
            back.paste(c_im,(20*i,0))
        return back

    attn_map = link_attn(dec)
    B,L,T = attn_map.shape
    
    inp = tensor2image(inp)
    pred = tensor2image(pred)
    label = label.int().cpu().tolist()
    
    for bi in range(B):
        print('    Drawing G Attn {}'.format(bi+st*B))
        t_line = num2char(label[bi], converter)
        len_t = len(t_line)+1
        back = Image.new('RGB', (len_t*20+32,18*32), (255,255,255))
        inp_im = inp[bi]
        # inp_im = 250*(inp_im-np.min(inp_im))/(np.max(inp_im)-np.min(inp_im))
        inp_im = np.clip(inp_im / 0.0078125 + 128,0,255)
        inp_im = np.uint8(inp_im)
        inp_im = Image.fromarray(inp_im)
        back.paste(inp_im,(0,0))

        pred_im = pred[bi]
        # pred_im = 250*(pred_im-np.min(pred_im))/(np.max(pred_im)-np.min(pred_im))
        pred_im = np.clip(pred_im / 0.0078125 + 128,0,255)
        pred_im_copy = pred_im.copy()
        pred_im_copy = np.uint8(pred_im_copy)
        pred_im_copy = Image.fromarray(pred_im_copy)   
        back.paste(pred_im_copy,(35,0))
        pred_im = pred_im * 0.5
        back.paste(inp_im,(0,0))

        for li in range(L):
            attn = attn_map[bi,li,:]
            attn_text = make_line(t_line, attn)
            h = li//4
            w = li%4
            pred_im_copy = pred_im.copy()
            pred_im_copy[h*8:(h+1)*8,w*8:(w+1)*8] = pred_im_copy[h*8:(h+1)*8,w*8:(w+1)*8]*2
            pred_im_copy = np.uint8(pred_im_copy)
            pred_paste = Image.fromarray(pred_im_copy)        
            back.paste(pred_paste, (0,35 + li*32))
            back.paste(attn_text, (35,35+li*32))
        
        back.save(savepath + '/visual_G_{}.jpg'.format(bi+st*B))

def attn_map_str2img(inp, dec, pred, label, st, config, converter):
    font = pygame.font.Font('lib/core/a.ttf', 26)
    attn_map = link_attn(dec)
    B,L,T = attn_map.shape
    pred = tensor2image(pred)
    label = label.int().cpu().tolist()

    for bi in range(B):
        print('    Drawing G Map {}'.format(bi+st*B))
        p_line = num2char(label[bi], converter) + ['B']
        lens = len(p_line)
        back = Image.new('RGB', (lens*32+35, 16*32+35), (255,255,255))

        block = np.ones((8,8,3),dtype=np.float32) * 50 
        block[:,:,1:] *= -1
        im = pred[bi]
        src = 255*(im-np.min(im))/(np.max(im)-np.min(im))
        src = np.expand_dims(src, axis=-1).repeat(3, axis=-1)
        for i in range(16):
            im = src.copy()
            x = i//4
            y = i%4
            im[x*8:(x+1)*8,y*8:(y+1)*8] += block
            im = np.clip(im,0,255)
            im = Image.fromarray(np.uint8(im))
            back.paste(im,(0,i*32))

        for i,c in enumerate(p_line):
            im = make_a_char(font, c, (0,0,0), (255,255,255))
            back.paste(im, (35+i*32+3,16*32+3))

        attn = attn_map[bi,:,:lens]
        attn = attn.repeat(32,axis=0).repeat(32,axis=1)
        attn = np.expand_dims(attn, -1)
        attn = 255 - attn.repeat(2,axis=2) * 255
        attn = np.clip(attn,0,255)
        red = np.ones((16*32,lens*32,1))*255
        attn = np.concatenate([red, attn], axis=-1)
        attn = np.uint8(attn)
        attn = Image.fromarray(attn)
        back.paste(attn, (35,0))
        back.save(config.MODEL.RESULT_SAVE_PATH + '/map_G_{}.jpg'.format(bi+st*B))
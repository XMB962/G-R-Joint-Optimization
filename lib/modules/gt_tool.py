import os
read = open('gtlabel_9800.txt','r',encoding='utf-8-sig').readlines()
dct = []
for line in read:
    char = line.strip('\n').split()[1]
    if char not in dct:
        dct.append(char)
    else:
        print(char,'appear twice')
dct = {line.strip('\n').split()[1]:int(line.split()[0]) for line in read}
import sys
class CODE(object):
    def __init__(self, dct):
        self.old_char = ''.join([chr(i) for i in range(65281,65375)])
        #！＂＃＄％＆＇（）＊＋，－．／０１２３４５６７８９：；＜＝＞？＠ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ［＼］＾＿｀ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ｛｜｝～
        self.new_char = ''.join([chr(i) for i in range(33,127)]) 
        #!"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~

        self.old_char += '【】〔〕〖〗﹝﹞'
        self.new_char += '[][][][]'

        self.old_char += '﹑丶│丨︳∣ㅣ¦〝〞″'
        self.new_char += '、、||||||"""'

        self.old_char += '∧ʌ︰∶Ұ¥'
        self.new_char += 'ΛΛ::￥￥'

        self.old_char += '—–━―─﹣『』'
        self.new_char += '------「」'

        self.old_char += '‧ㆍ•∙▪﹒・′´ˊ'
        self.new_char += '·······\'\'\''

        self.old_char += '‹﹤﹥›╱╲'
        self.new_char += '<<>>/\\'

        self.old_char += '○〇О'
        self.new_char += 'o0O'

        self.old_char += '□☆◇￡∫®ℰº✕℅ˋ～×﹐﹖﹔'
        self.new_char += '■★◆£ʃⓇε°X%`~x，?;' 

        self.old_char += '﹗﹢'
        self.new_char += '!+' 

        keys = list(dct.keys())
        old_char = self.old_char
        for c in old_char:
            if c in keys:
                idx = self.old_char.find(c)
                self.old_char = self.old_char[:idx] + self.old_char[idx+1:]
                self.new_char = self.new_char[:idx] + self.new_char[idx+1:]
                print(c,'is existed')

        print('lens char new/old:',len(self.new_char),len(self.old_char))
        if len(self.new_char)!=len(self.old_char):
            sys.exit()

        self.old_word = [chr(i) for i in range(9332,9352)] #(1)
        self.old_word += [chr(i) for i in range(9352,9372)] #1.
        self.old_word += [chr(i) for i in range(9372,9398)] #(a)
        self.old_word += [chr(i) for i in range(8544, 8556)] #III
        self.old_word += ['…', '┅', '¼', '½', '¾', '№', 'N°', '㎡']
        self.old_word += ['㎎', '㎏', '㎜', '㎝', '㎞', '㏄']
        self.old_word += ['»']
        
        self.new_word = ['({})'.format(i) for i in range(1,21)]
        self.new_word += ['{}.'.format(i) for i in range(1,21)]
        self.new_word += ['({})'.format(chr(i+97)) for i in range(26)]
        self.new_word += ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII']
        self.new_word += ['...', '...', '1/4', '1/2', '3/4', 'NO.', 'NO.' ,'m²']
        self.new_word += ['mg', 'kg', 'mm', 'cm', 'km', 'cc']
        self.new_word += ['>>']

        print('lens word new/old:',len(self.new_word),len(self.old_word))
        if len(self.new_word)!=len(self.old_word):
            sys.exit()

        self.delete = ['なハッピーホしんいちょくどぅリマシサジデがーてぉにき' ,'┌┐', '￣', 'ΦΟɔ','﹃','\xad','⑪', '丄', 'ㄱ', '︽', '︾', '≦', '๑', 'ㄑ', '㊣', '⑳', '⺕', '∟', '¬', '︿']
        self.delete += [chr(i) for i in range(7680,8704)] # Ḁ ḁ Ḃ ḃ
        self.delete += [chr(i) for i in range(1025,1106)] # 俄文
        self.delete += [chr(i) for i in range(192,688)] # À Á Â Ã
        self.delete += [chr(i) for i in range(9471,10570)] #表情、制表符、箭头
        self.delete = ''.join([c for c in self.delete])

        delete = self.delete
        for c in delete:
            if c in keys:
                idx = self.delete.find(c)
                self.delete = self.delete[:idx] + self.delete[idx+1:]
                print(c,' should not delete')

    def go(self,inp):
        for o,n in zip(list(self.old_char),list(self.new_char)):
            if o in inp:
                inp = inp.replace(o,n)
        for o,n in zip(self.old_word,self.new_word):
            if o in inp:
                inp = inp.replace(o,n)
        for c in inp:
            if c in self.delete:
                return None
        return inp 
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 15:13:34 2018

@author: Administrator
"""
import numpy as np
from PIL import Image

binaryCode=[    '00001', '00100', '00110', '01100',
                '01110', '10000', '10011', '10101',
                '11010', '11101', '00011', '00101', 
                '01001', '01011', '01111', '10010',
                '10100', '11001', '11100', '11110',
                '00000'
                ]
text='PQRYWTMNVELHSFCIKADG'

def rule(n):
    b = list(bin(n))
    b = b[2:]
    k = 8 - len(b)
    s = []
    while k > 0:
        s.append('0')
        k -= 1
    for i in b:
        s.append(i)
    
    return s

# 对序列0-1表示的序列sequence按规则n(0<=n<=255)演化
# 返回演化后的start~end步的序列
def evolve(sequence, n, start, end):
    p = ['111','110','101','100','011','010','001','000']
    r = rule(n)
    ca = []
    ca.append(sequence)
    es = sequence
    
    for i in range(end):
        seq = es[-1] + es + es[0]
        es = ''
        for k in range( len(seq) - 2):
            nb = seq[k:k+3]
            indx = p.index(nb)
            es = es + r[indx]
        if i >= start:
            ca.append(es)
    if start == 0:
        return ca 
    else:
        return ca[1:]

def CAMatrix(protseq, n, start, end):
    bs = ''
    for aa in protseq:
        try:
            i = text.index(aa)
            bs = bs + binaryCode[i]
        except ValueError:
            bs = bs + '00000' 
        
    ca = evolve(bs, n, start, end)   
    row = len(ca)
    col = len(ca[0])

    mat = np.ndarray( (row, col))
    for i in range(row):
        for j in range(col):
            mat[i,j] = float( ca[i][j])
    return mat

def CAImage(ca):
    row = len( ca)
    col = len( ca[0])     
    image = np.ndarray((row,col), dtype=np.int)
    for i in range( row):
        for j in range( col):
            image[i,j] = int( ca[i][j])
    img = Image.fromarray(image*255, 'L')
    return img
 
def CAArray(ca, shape):
    row = len( ca)
    col = len( ca[0])     
    image = np.ndarray((row,col), dtype=np.int)
    for i in range( row):
        for j in range( col):
            image[i,j] = int( ca[i][j])
    img = Image.fromarray(image, 'L')
    img = img.resize(shape)
    return img

def generateCAArrayOfSeq(protseq,r,start,end,shape):
    bs = ''
    for aa in protseq:
        try:
            i = text.index(aa)
            bs = bs + binaryCode[i]
        except ValueError:
            bs = bs + '00000'
    ca = evolve(bs,r,start,end)
    img = CAArray(ca,shape)
    return np.array(img)

# 从蛋白质序列生成元胞自动机图像
# protseq: 蛋白质序列
# r: 元胞自动机演化规则
# start, end：图像仅保存从start到end之间的演化结果
def generateCAImageOfSeq(protseq,r,start,end):
    bs = ''
    for aa in protseq:
        try:
            i = text.index(aa)
            bs = bs + binaryCode[i]
        except ValueError:
            bs = bs + '00000'
    ca = evolve(bs,r,start,end)
    img = CAImage(ca)
    return img


def createCAImageFileOfSeq(protseq, r, start, end, imageFile):
    img = generateCAImageOfSeq(protseq,r,start,end)
    img.save(imageFile,'jpeg')
'''    
def main():
    s = 'MRBGSHHHHHHTDPHASSVPLEWPLSSQSGSYELRIEVQPKPHHRAHYETEGSRGAVKAPTGGHPVVQLHGYMENKPLGLQIFIGTADERILKPHAFYQVHRITGKTVTTTSYEKIVGNTKVLEIPLEPKNNMRATIDCAGILKLRNADIELRKGETDIGRKNTRVRLVFRVHIPESSGRIVSLQTASNPIECSQRSAHELPMVERQDTDSCLVYGGQQMILTGQNFTSESKVVFTEKTTDGQQIWEMEATVDKDKSQPNMLFVEIPEYRNKHIRTPVKVNFYVINGKRKRSQPQHFTYHPV'
    print(s)
    print(generateCAImageOfSeq(s,84,100,200))

if __name__ == "__main__":
    main()
'''
s = 'MRBGSHHHHHHTDPHASSVPLEWPLSSQSGSYELRIEVQPKPHHRAHYETEGSRGAVKAPTGGHPVVQLHGYMENKPLGLQIFIGTADERILKPHAFYQVHRITGKTVTTTSYEKIVGNTKVLEIPLEPKNNMRATIDCAGILKLRNADIELRKGETDIGRKNTRVRLVFRVHIPESSGRIVSLQTASNPIECSQRSAHELPMVERQDTDSCLVYGGQQMILTGQNFTSESKVVFTEKTTDGQQIWEMEATVDKDKSQPNMLFVEIPEYRNKHIRTPVKVNFYVINGKRKRSQPQHFTYHPV'
m = generateCAArrayOfSeq(s,84,100,200,(28,28))        
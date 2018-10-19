# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 15:13:34 2018

@author: Administrator
"""
import numpy as np
from PIL import Image
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
    return ca       
    

def CAImage(ca):
    row = len( ca)
    col = len( ca[0])     
    image = np.ndarray((row,col), dtype=np.int)
    for i in range( row):
        for j in range( col):
            image[i,j] = int( ca[i][j])
    img = Image.fromarray(image*255, 'L')
    return img
           
            
def main():
    s = '101110100'
    ca = evolve(s,84,0,10)
    print(ca)
    ca = evolve(s,84,5,9)
    print(ca)

if __name__ == "__main__":
    main()
        
        
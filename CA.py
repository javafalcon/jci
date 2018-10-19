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

# 对序列sequence按规则n(0<=n<=255)演化m次
def evolve(sequence, n, m):
    p = ['111','110','101','100','011','010','001','000']
    r = rule(n)
    ca = []
    ca.append(sequence)
    es = sequence
    for i in range(m):
        seq = es[-1] + es + es[0]
        es = ''
        for k in range( len(seq) - 2):
            nb = seq[k:k+3]
            indx = p.index(nb)
            es = es + r[indx]
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
    ca = evolve(s,84,4)
    img = CAImage(ca)
    img.save('t.jpg','jpeg')
    

if __name__ == "__main__":
    main()
        
        
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 13:21:41 2018

@author: Administrator
"""
from CA import *

binaryCode=[    '00001', '00100', '00110', '01100',
                '01110', '10000', '10011', '10101',
                '11010', '11101', '00011', '00101', 
                '01001', '01011', '01111', '10010',
                '10100', '11001', '11100', '11110',
                '00000'
                ]
text='PQRYWTMNVELHSFCIKADGX'
# 从蛋白质序列生成元胞自动机图像
# protseq: 蛋白质序列
# r: 元胞自动机演化规则
# start, end：图像仅保存从start到end之间的演化结果
def generateCAImageFromSeq(protseq,r,start,end):
    bs = ''
    for aa in protseq:
        i = text.index(aa)
        bs = bs + binaryCode[i]
    return bs

def main():
    protseq='APRY'
    print(generateCAImageFromSeq(protseq))
    
if "__name__" == "__main__":
    main()
    
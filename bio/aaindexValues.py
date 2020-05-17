# -*- coding: utf-8 -*-
"""
Created on Sun May  3 16:34:55 2020

@author: lwzjc
"""
import numpy as np

AminoAcids = 'ARNDCQEGHILKMFPSTWYV'
def aaindex1Values():
    AAValues=[]
    newlineFlag = False
    addValueFlag = False
    t = []
    with open('E:\\Repoes\\jci\\bio\\aaindex1') as fr:
        lines = fr.readlines()
        for line in lines:
            if line.startswith('I'):
                newlineFlag = True
                addValueFlag = True
                t = []
                continue
            elif line.startswith('//'):
                newlineFlag = False
                if addValueFlag:
                    AAValues.append(t)
     
            if newlineFlag:
                line = line.strip()
                if 'NA' in line:
                    addValueFlag = False
                    continue
                vals = line.split()
                for v in vals:
                    t.append(eval(v))
    return np.array(AAValues)      

if __name__ == '__main__':
    aavals = aaindex1Values()
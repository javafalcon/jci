# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 22:23:44 2018

@author: falcon1
"""

import numpy as np
from Bio import SeqIO
import json
import re

# 对矩阵进行归一化
# 每行是一个样本，每列是一个特征
# 对每一列特征进行归一化： (f-min)/(max-min)

amino_acids = 'PQRYWTMNVELHSFCIKADG'

def maxminnorm(array):
    maxcols=array.max(axis=0)
    mincols=array.min(axis=0)
    data_shape = array.shape
    data_rows = data_shape[0]
    data_cols = data_shape[1]
    t=np.empty((data_rows,data_cols))
    for i in range(data_cols):
        t[:,i]=(array[:,i]-mincols[i])/(maxcols[i]-mincols[i])
    return t

# 加载来自hmmer profil的数据
# hmmer_profil文件是一个json格式的文件，文件存放的数据以字典表示
# key是蛋白质序列的ID，value是使用jackerhmmer搜索数据库得到的进化矩阵
# num_aas>0表示从序列中截取的氨基酸个数
def load_hmm_prof( hmm_profil_json, num_aas):
    fr = open(hmm_profil_json)
    prof = json.load(fr)
    keys = prof.keys()
    num_rows = len(keys)
    N = num_aas * 20
    X = np.ndarray([num_rows, N])    
    
    k = 0
    
    for key in keys:
        ary = prof[key]
        tm = np.array(ary).reshape([-1,20])
        c = len(ary)
        if c < N:
            tm = maxminnorm(tm)# 归一化
            X[k][:c] = tm.reshape(c)
            X[k][c:] = 0
        elif c == N:
            tm = maxminnorm(tm)# 归一化
            X[k] = tm.reshape(c)
        else:
            t = tm[:50,:]
            t = maxminnorm(t)# 归一化
            X[k] = t.reshape(N)
        k += 1
        
    fr.close()
        
    return X

def seqAAOneHot(seq, start=0, length=0):
    """
    表达序列的AA One-Hot 矩阵表示
    
    Parameters:
    ___________
    seq: string
         sequence of amino acid
    start,length: int
         if start > 0, length也需大于0，取序列切片seq[start:start+length]
         if start < 0, length也需小于0，取序列切片seq[start+length+1:start+1]
         if start取默认值0，length>0,取序列切片seq[:,length]
         if start取默认值0，length<0,取序列切片seq[length:]
    Returns:
    __________
    numpy.ndarry
    """
    seq = re.sub('[XZUB]',"",seq)
    seq = seq.strip()
    if length == 0:
        length = len(seq) - start
        
    X = np.zeros((abs(length), 20))
    
    if start > 0 and length > 0:
        s = seq[start: start+length]
        for i in range(len(s)):
            j = amino_acids.index(s[i])
            X[i][j] = 1
    elif start < 0 and length < 0:
        s = seq[start+length+1:start+1]
        for i in range(-len(s), 0):
            j = amino_acids.index(s[i])
            X[i][j] = 1
    elif start == 0 and length > 0:
        s = seq[:length]
        for i in range(len(s)):
            j = amino_acids.index(s[i])
            X[i][j] = 1
    elif start == 0 and length < 0:
        s = seq[length:]
        for i in range(-len(s),0):
            j = amino_acids.index(s[i])
            X[i][j] = 1
            
    return X

def seqDAA(seq, start=0, end=0):
    seq = re.sub('[XZUB]',"",seq)
    seq = seq.strip()
    if end == 0:
        s = seq[start:]
    else:
        s = seq[start:end]
    
    X = np.zeros((20,20))
    for i in range(20):
        for j in range(20):
            X[i][j] = s.count("".join([amino_acids[i], amino_acids[j]]))
    
    return X/np.sum(X)
                
    
def AAOneHot(fastafile, num_aas):
    #files=[r'E:\Repoes\AMPnet\data\benchmark\AMPs_50.fasta',r'E:\Repoes\AMPnet\data\benchmark\notAMPs_50.fasta']
    text='PQRYWTMNVELHSFCIKADG'
    seq_records = list(SeqIO.parse(fastafile, 'fasta'))
    
    N = num_aas * 20
    num_rows = len(seq_records)
    X = np.ndarray((num_rows,N))
     
    k = 0
    
    for seq_record in seq_records:
        seq = str(seq_record.seq)
        seq = re.sub('[XZUB]',"",seq)
        seq = seq.strip()
        
        c = len(seq)
        m = np.zeros((len(seq),20))
        for i in range(c):
            j = text.index(seq[i])
            m[i][j] = 1
       
        m = m.reshape((1,-1))
        
        # 只截取蛋白质序列前num_aas个aa，不足的补0
        c = c*20
        if c < N:
            X[k][:c] = m[0]
            X[k][c:] = 0
        elif c == N:
            X[k] = m[0]
        else:
            X[k] = m[0][:N] 
        k += 1
    return X

# 氨基酸物化属性ONEHOT编码
def AAPhyChemOneHot(fastafile, num_aas):
    #files=[r'E:\Repoes\AMPnet\data\benchmark\AMPs_50.fasta',
    #       r'E:\Repoes\AMPnet\data\benchmark\notAMPs_50.fasta']
    seq_records = list(SeqIO.parse(fastafile, 'fasta'))
    
    N = num_aas * 20
    num_rows = len(seq_records)
    X = np.ndarray((num_rows,N))
     
    k = 0
    phychemDict={}
    phychemDict["alcohol"]=("S","T")# 有乙醇基
    phychemDict["aliphatic"]=("I","L","V")# 脂肪族
    phychemDict["aromatic"]=("F","H","W","Y")# 芳香族
    phychemDict["charged"]=("D","E","H","K","R")# 带电性
    phychemDict["positive"]=("K","H","R")# 带正电
    phychemDict["negative"]=("D","E")# 带负电
    phychemDict["polar"]=("A","L","I","P","F","W","M")# 非极性
    phychemDict["small"]=("A","C","D","G","N","P","S","T","V")# 小分子
    phychemDict["turnlike"]=("A","C","D","E","G","H","K","N","Q","R","S","T")
    phychemDict["hydrophobic"]=("A","F","I","L","M","P","V","W","Y")# 疏水
    phychemDict["asa"]=("A","N","D","C","P","S","T","G","V")# 可溶解表面积低于平均值
    phychemDict["pr"]=("F","Y","W")# 在紫外区有光吸收能力
    
    keys = phychemDict.keys()
    
    
    for seq_record in seq_records:
        seq = str(seq_record.seq)
        seq = re.sub('[XZUB]',"",seq)
        c = len(seq)
        m = np.zeros((len(seq),20))
        for i in range(c):
            j = 0
            for key in keys:
                val = phychemDict[key]
                if seq[i] in val:
                    m[i][j] = 1
                j += 1
            
        m = m.reshape((1,-1))
        #print("in {},{}:{}".format(file,seq_record.id,m.shape))
        # 只截取蛋白质序列前50个aa，不足的补0
        c = c*20
        if c < N:
            X[k][:c] = m[0]
            X[k][c:] = 0
        elif c == N:
            X[k] = m[0]
        else:
            X[k] = m[0][:N] 
        k += 1
    return X

def DAA(fastafile):
    """
    蛋白质序列两连体表示矩阵
    
    Parameters
    __________
    fastafile: string
               the fasta file's name of sequences
    
    Returns
    _________
    numpy.ndarray
          the douple amino acid composition features of sequences
    """
    #amino_acids = 'PQRYWTMNVELHSFCIKADG'
    
    seq_records = list(SeqIO.parse(fastafile, 'fasta'))
    num_seqs = len(seq_records)
    
    X = np.ndarray([num_seqs,400])
    k = 0
    
    for seq_record in SeqIO.parse(fastafile,'fasta'):
        x = np.ndarray([20,20])
        seq = str(seq_record.seq)
        seq = re.sub('[XZUB]',"",seq)
        
        for i in range(20):
            for j in range(20):
                x[i][j] = seq.count("".join([amino_acids[i], amino_acids[j]]))
        
        x = x.reshape([1,400])
        x = x/np.sum(x)
        
        X[k] = x
        k = k + 1
        
    return X

"""
return code of amino acid
"""
def AACode(amino_acid,codeType,norm=False):
    numCode = []
    normNumCode = []
    #MolecularWeight  
    numCode.append([89.09, 174.20, 132.12, 133.10, 121.15, 146.15, 147.13, 75.07, 155.16, 131.17,
        131.17, 146.19, 149.21, 165.19, 115.13, 105.09, 119.12, 204.24, 181.19, 117.15])
    #norm_molweig = 
    normNumCode.append([-1.5490, 1.2084, -0.1549, -0.1231, -0.5103, 0.2997, 0.3314, -2.0032, 0.5916, -0.1857, -0.1857,
        0.3009, 0.3988, 0.9165, -0.7053, -1.0306, -0.5761, 2.1817, 1.4349, -0.6399])
    #Hydrophobicity  
    numCode.append([0.87, 0.85, 0.09, 0.66, 1.52, 0, 0.67, 0.1, 0.87, 3.15,
        2.17, 1.64, 1.67, 2.87, 2.77, 0.07, 0.07, 3.77, 2.67, 1.87])
    #norm_hydrophobicity  
    normNumCode.append([-0.4669, -0.4839, -1.1320, -0.6459, 0.0874, -1.2087, -0.6374, -1.1234, -0.4669, 1.4773,
        0.6417, 0.1897, 0.2153, 1.2385, 1.1533, -1.1490, -1.1490, 2.0060, 1.0680, 0.3858])
    #pk1 
    numCode.append([2.35, 2.18, 2.18, 1.88, 1.71, 2.17, 2.19, 2.34, 1.78, 2.32,
        2.36, 2.20, 2.28, 2.58, 1.99, 2.21, 2.15, 2.38, 2.20, 2.29])
    #norm_pk1 
    normNumCode.append([0.7764, -0.0333, -0.0333, -1.4623, -2.2721, -0.0810, 0.0143, 0.7288, -1.9387, 0.6335, 0.8240, 0.0619,
        0.4430, 1.8720, -0.9384, 0.1096, -0.1762, 0.9193, 0.0619, 0.4906])
    #pk2 
    numCode.append([9.87, 9.09, 9.09, 9.60, 10.78, 9.13, 9.67, 9.60, 8.97, 9.76,
        9.60, 8.90, 9.21, 9.24, 10.60, 9.15, 9.12, 9.39, 9.11, 9.74])
    #norm_pk2 
    normNumCode.append([0.7692, -0.7732, -0.7732, 0.2353, 2.5687, -0.694, 0.3737, 0.2353, -1.0105, 0.5517, 0.2353, -1.1489,
        -0.5359, -0.4766, 2.2128, -0.6545, -0.7139, -0.1799, -0.7336, 0.5122])
    #PI 
    numCode.append([6.11, 10.76, 10.76, 2.98, 5.02, 5.65, 3.08, 6.06, 7.64, 6.04,
        6.04, 9.47, 5.74, 5.91, 6.30, 5.68, 5.60, 5.88, 5.63, 6.02])
    #norm_PI 
    normNumCode.append([-0.1033, 2.2014, 2.2014, -1.6547, -0.6436, -0.3313, -1.6051, -0.1281, 0.6550, -0.1380, -0.1380, 1.5620,
        -0.2867, -0.2025, -0.0092, -0.3165, -0.3561, -0.2173, -0.3412, -0.1479])
    text = "ARNDCQEGHILKMFPSTWYV"
    TYPE = ['MolecularWeight','Hydrophobicity','PK1','PK2','PI']
    t = TYPE.index(codeType)
    k = text.index(amino_acid)
    if norm:
        return normNumCode[t][k]
    else:
        return numCode[t][k]
        
"""
calculate the pseudo amino acid composition
and the PseAACs are generated by grey mode.
If model=1,by GM(1,1); else model=2, by GM(2,1)
""" 
from greymodel import GMParam
import math
def greyPseAAC(seq, codeTypes, model=1):
    seq = re.sub('[XZUB]',"",seq)
    seq = seq.strip()
    seq = seq.upper()
    n = len(seq)
    pseaac = []
    #The first 20 elements in x are the 20 amino acids' frequence
    for aa in amino_acids:
        pseaac.append(seq.count(aa)/n)
    
    
    for codeType in codeTypes:
        x = []
        for a in seq:
            e = math.exp(-AACode(a,codeType,True))
            x.append(1/(1+e))
        a,b = GMParam(x,model)
        pseaac.append(abs(a))
        pseaac.append(abs(b))
    
    return pseaac
                       
def main():
    codeTypes = ['MolecularWeight','Hydrophobicity','PK1','PK2','PI']
    seq = 'SLFEQLGGQAAVQAVTAQFYANIQADATVATFFNGIDMPNQTNKTAAFLCAALGGPNAWTGRNLKEVHAN\
MGVSNAQFTTVIGHLRSALTGAGVAAALVEQTVAVAETVRGDVVTV'
    v = greyPseAAC(seq,codeTypes)
    print(v)
    
if __name__ == "__main__":
    main()
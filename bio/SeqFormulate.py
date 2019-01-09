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

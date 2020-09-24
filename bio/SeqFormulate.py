# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 22:23:44 2018

@author: falcon1
"""

import numpy as np
from Bio import SeqIO
import json
import re
import math
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

def seqAAOneHot(seq, start=0, length=0, Xcode=False):
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
    if not Xcode:
        seq = re.sub('[XZUB]',"",seq)
        seq = seq.strip()
    if length == 0:
        length = len(seq) - start
        
    X = np.zeros((abs(length), 20))
    
    if start > 0 and length > 0:
        s = seq[start: start+length]
        for i in range(len(s)):
            try:
                j = amino_acids.index(s[i])
                X[i][j] = 1
            except ValueError:
                X[i] = 0.05 * np.ones((20,))              
            
    elif start < 0 and length < 0:
        s = seq[start+length+1:start+1]
        for i in range(-len(s), 0):
            try:
                j = amino_acids.index(s[i])
                X[i][j] = 1
            except ValueError:
                X[i] = 0.05 * np.ones((20,))
    elif start == 0 and length > 0:
        s = seq[:length]
        for i in range(len(s)):
            try:
                j = amino_acids.index(s[i])
                X[i][j] = 1
            except ValueError:
                X[i] = 0.05 * np.ones((20,))
    elif start == 0 and length < 0:
        s = seq[length:]
        for i in range(-len(s),0):
            try:    
                j = amino_acids.index(s[i])
                X[i][j] = 1
            except ValueError:
                X[i] = 0.05 * np.ones((20,))
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
    numCode = {}
    normNumCode = {}
    #MolecularWeight  
    numCode["MolecularWeight"] = [89.09, 174.20, 132.12, 133.10, 121.15, 146.15, 147.13, 75.07, 155.16, 131.17,
        131.17, 146.19, 149.21, 165.19, 115.13, 105.09, 119.12, 204.24, 181.19, 117.15]
    #norm_molweig  
    normNumCode["MolecularWeight"] = [-1.5490, 1.2084, -0.1549, -0.1231, -0.5103, 0.2997, 0.3314, -2.0032, 0.5916, -0.1857, -0.1857,
        0.3009, 0.3988, 0.9165, -0.7053, -1.0306, -0.5761, 2.1817, 1.4349, -0.6399]
    #Hydrophobicity  
    numCode["Hydrophobicity"] = [0.87, 0.85, 0.09, 0.66, 1.52, 0, 0.67, 0.1, 0.87, 3.15,
        2.17, 1.64, 1.67, 2.87, 2.77, 0.07, 0.07, 3.77, 2.67, 1.87]
    #norm_hydrophobicity  
    normNumCode["Hydrophobicity"] = [-0.4669, -0.4839, -1.1320, -0.6459, 0.0874, -1.2087, -0.6374, -1.1234, -0.4669, 1.4773,
        0.6417, 0.1897, 0.2153, 1.2385, 1.1533, -1.1490, -1.1490, 2.0060, 1.0680, 0.3858]
    #pk1 
    numCode["PK1"] = [2.35, 2.18, 2.18, 1.88, 1.71, 2.17, 2.19, 2.34, 1.78, 2.32,
        2.36, 2.20, 2.28, 2.58, 1.99, 2.21, 2.15, 2.38, 2.20, 2.29]
    #norm_pk1 
    normNumCode["PK1"] = [0.7764, -0.0333, -0.0333, -1.4623, -2.2721, -0.0810, 0.0143, 0.7288, -1.9387, 0.6335, 0.8240, 0.0619,
        0.4430, 1.8720, -0.9384, 0.1096, -0.1762, 0.9193, 0.0619, 0.4906]
    #pk2 
    numCode["PK2"] = [9.87, 9.09, 9.09, 9.60, 10.78, 9.13, 9.67, 9.60, 8.97, 9.76,
        9.60, 8.90, 9.21, 9.24, 10.60, 9.15, 9.12, 9.39, 9.11, 9.74]
    #norm_pk2 
    normNumCode["PK2"] = [0.7692, -0.7732, -0.7732, 0.2353, 2.5687, -0.694, 0.3737, 0.2353, -1.0105, 0.5517, 0.2353, -1.1489,
        -0.5359, -0.4766, 2.2128, -0.6545, -0.7139, -0.1799, -0.7336, 0.5122]
    #PI 
    numCode["PI"] = [6.11, 10.76, 10.76, 2.98, 5.02, 5.65, 3.08, 6.06, 7.64, 6.04,
        6.04, 9.47, 5.74, 5.91, 6.30, 5.68, 5.60, 5.88, 5.63, 6.02]
    #norm_PI 
    normNumCode["PI"] = [-0.1033, 2.2014, 2.2014, -1.6547, -0.6436, -0.3313, -1.6051, -0.1281, 0.6550, -0.1380, -0.1380, 1.5620,
        -0.2867, -0.2025, -0.0092, -0.3165, -0.3561, -0.2173, -0.3412, -0.1479]
    # Accessible surface area
    numCode["ASE"] = [93.7, 250.4, 146.3, 142.6, 135.2, 177.7, 182.9, 52.6, 188.1, 182.2, 173.7, 215.2,
                      197.6, 228.6,  0., 109.5, 142.1, 271.6, 239.9, 157.2]
    text = "ARNDCQEGHILKMFPSTWYV"
    if norm:
        t = normNumCode[codeType]       
    else:
        t = numCode[codeType]
    i = text.index(amino_acid)
    return t[i]
        
"""
calculate the pseudo amino acid composition
and the PseAACs are generated by grey mode.
If model=1,by GM(1,1); else model=2, by GM(2,1)
""" 
from greymodel import GMParam
import math
def greyPseAAC(seq, codeTypes, weight=None, model=1, norm=False):
    seq = seq.upper()
    seq = re.sub('[#XZUB]',"",seq)
    seq = seq.strip()
    
    n = len(seq)
    pseaac = []
    #The first 20 elements in x are the 20 amino acids' frequence
    for aa in amino_acids:
        pseaac.append(seq.count(aa)/n)
    
    
    for codeType in codeTypes:
        x = []
        for a in seq:
            if norm:
                e = math.exp(-AACode(a, codeType, norm))
                x.append(1/(1+e))
            else:
                x.append(AACode(a, codeType, norm))
        gp = GMParam(x,model)
        for p in gp:
            pseaac.append(abs(p))
    
    if weight is not None:
        pseaac = pseaac * weight
    return pseaac


def chaosGraph_aminoacids(seq, width=30, hight=30, norm=False):
    # 非极性且疏水，极性且中性；酸性，碱性
    aa = [['A','V','L','I','P','G','W','F','M'],['Q','S','T','C','N','Y'],['D','E'],['K','R','H']]  
    seq = seq.upper()
    seq = re.sub('[XZUB]','',seq)
    seq = seq.strip() 
    g = np.zeros(shape=(width, hight))
    i,j = width//2,hight//2
    for c in seq:
        if c in aa[0]:
            x,y = 0,0
        elif c in aa[1]:
            x,y = width,0
        elif c in aa[2]:
            x,y = width, hight
        elif c in aa[3]:
            x,y = 0,hight
            
        tx, ty = (i + x)//2, (j + y)//2
        g[tx,ty] = g[tx,ty] + 1
        i,j = tx,ty

    if norm:
        return g/len(seq)
    else:
        return g


    
# build chaos graph of protein sequence by double amino-acids
def DAA_chaosGraph(sequences:list):    
    AminoAcids = 'ARNDCQEGHILKMFPSTWYVX'
    AA={}
    for a in AminoAcids:
        for b in AminoAcids:
            AA[a+b] = (AminoAcids.index(a)+0.5, AminoAcids.index(b)+0.5)
    
    X = []     
    regexp = re.compile('[^ARNDCQEGHILKMFPSTWYV]')
    for seq in sequences:
        seq = regexp.sub('X', seq)
        x = []
        x.append((0,0))
        for i in range(len(seq) - 1):
            p = AA[seq[i:i+2]]
            x.append( ((x[i][0] + p[0]) / 2, (x[i][1] + p[1]) / 2))
        t = np.zeros(shape=(21,21))
        for i in range(1, len(x)):
            t[math.floor(x[i][0]), math.floor(x[i][1])] += 1 
        t /= np.sum(t)
        X.append(t)
     
    return np.array(X)

def cor_chaosGraph(sequences:list, r=5):
    AminoAcids = 'ARNDCQEGHILKMFPSTWYVX'
    AA={}
    for a in AminoAcids:
        for b in AminoAcids:
            AA[a+b] = (AminoAcids.index(a)+0.5, AminoAcids.index(b)+0.5)
    
    X = np.zeros((len(sequences),21,21,r))     
    regexp = re.compile('[^ARNDCQEGHILKMFPSTWYV]')
    
    for k in range(len(sequences)):
        seq = regexp.sub('X', sequences[k])
        for m in range(1,r+1):
            x = []
            x.append((0,0))
            for i in range(len(seq) - m):
                p = AA[seq[i]+seq[i+m]]
                x.append( ( (x[i][0] + p[0]) / 2, (x[i][1] + p[1]) / 2))
            t = np.zeros(shape=(21,21))
            for j in range(1, len(x)):
                t[math.floor(x[j][0]), math.floor(x[j][1])] += 1
            t /= np.sum(t)
            
            X[k,:,:,m-1] = t
    
    return X

def corr_onehot(sequences, r=5):
    AminoAcids = 'ARNDCQEGHILKMFPSTWYVX'
    regexp = re.compile('[^ARNDCQEGHILKMFPSTWYV]')

    X = np.zeros((len(sequences), 21, 21, r))
    for k in range(len(sequences)):
        seq = regexp.sub('X', sequences[k])
        for m in range(1, r+1):
            x = np.zeros((21,21))
            for i in range(len(seq)-m):
                col = AminoAcids.index(seq[i])
                row = AminoAcids.index(seq[i+m])
                x[col][row] += 1
            x = x/np.sum(x)
            X[k, :, :, m-1] = x
        
    return X

def TAA_chaosGraph(sequences:list):
    AminoAcids = 'ARNDCQEGHILKMFPSTWYV'
    TAA={}  
    for a in AminoAcids:
        for b in AminoAcids:
            for c in AminoAcids:
                TAA[a+b+c] = (AminoAcids.index(a)+0.5, AminoAcids.index(b)+0.5, AminoAcids.index(c)+0.5)
    X =  []
    regexp = re.compile('[^ARNDCQEGHILKMFPSTWYV]')
    for seq in sequences:
        seq = regexp.sub('', seq)
        x = []
        x.append((0,0,0))
        for i in range(len(seq) - 2):
            p = TAA[seq[i:i+3]]
            x.append( ((x[i][0] + p[0]) / 2, (x[i][1] + p[1]) / 2, (x[i][2] + p[2]) / 2)) 
        t = np.zeros(shape=(20,20,20))
        for i in range(1, len(x)):
            t[math.floor(x[i][0]), math.floor(x[i][1]), math.floor(x[i][2])] += 1
        t /= np.sum(t)
        t = t.reshape((-1,20*20*20))
        X.append(t)
    X = np.array(X)   
    return X.reshape((-1, 100, 80))

def main():
    #codeTypes = ['MolecularWeight','Hydrophobicity','PK1','PK2','PI']
    seqs = ['SLXFEQLGGQAAVQAVTAQFYANIQADATVATFFNGIDMPNQTNKTAAFLCAALGGPNAWTGRNLKEVHAN\
MGVSNAQFTTVIGHLRSALTGAGVAAALVEQTVAVAETVRGDVVTV',
            'LFEQLGGQAAVQAVTAQFYANIQADATVATFFNGIDMPNQTNKTAAFLCAALGGPNAWTGRNLKEVHAN\
MGVSNAQFTTVIGHLRSALTGAGVAAALVEQTVAVAETVRGDVVTVS']
    """seq = 'MDLLAELQWRGLVNQTTDEDGLRKLLNEERVTLYCGFDPTADSLHIGHLATILTMRRFQQAGHRPIALVGGATGLI\
    GDPSGKKSERTLNAKETVEAWSARIKEQLGRFLDFEADGNPAKIKNNYDWIGPLDVITFLRDVGKHFSVNYMMAKESVQSRI\
    ETGISFTEFSYMMLQAYDFLRLYETEGCRLQIGGSDQWGNITAGLELIRKTKGEARAFGLTIPLVTKADGTKFGKTESGTIWL\
    DKEKTSPYEFYQFWINTDDRDVIRYLKYFTFLSKEEIEALEQELREAPEKRAAQKTLAEEVTKLVHGEEALRQAIRIS'"""
        
    X2 = corr_onehot(seqs,5)
    
    return X2
if __name__ == "__main__":
    X2 = main()
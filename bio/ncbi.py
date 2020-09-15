#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 10:06:49 2018

@author: weizhong
"""

from Bio.Blast.Applications import NcbipsiblastCommandline
from Bio import SeqIO
import scipy.io as sio
import csv
import os
import numpy as np
import math

blosum = sio.loadmat('e:/repoes/jci/bio/blosum.mat')
blosumMatrix = blosum['blosum62']
alphabet = 'ARNDCQEGHILKMFPSTWYVBZX*'

# generate the PSSM file of each protein in fastafile by psi-blast
def getPSSMFiles(fastafile,outfileprefix='',dbName='swissprot'):
    
    inputfile = 'input.fasta'
    
    for seq_record in SeqIO.parse(fastafile, 'fasta'):
        print('\r{} '.format(seq_record.id),end="")
        # psi-blast input file
        if os.path.exists(inputfile):
            os.remove( inputfile)
        SeqIO.write( seq_record, inputfile, 'fasta')
        
        # psi-blast output file
        pssmfile = "".join( (outfileprefix, seq_record.id, '.txt'))
        if os.path.exists(pssmfile):
            os.remove( pssmfile)
        
        # psi-blast
        psiblast_cline = NcbipsiblastCommandline( query = inputfile, db=dbName, evalue=0.001,
                                                 num_iterations=3, out_ascii_pssm=pssmfile)
        stdout,stderr=psiblast_cline()
        
        # If seq_record does not have pssm, generating it by blosum62 Matrix
        if not os.path.exists(pssmfile):
            print('\r{} does not have pssm'. format(seq_record.id))
            with open(pssmfile,'w') as pw:
                pw.writelines("  \n")
                pw.writelines("last position-specific scoring matrix computed, weighted \n")
                pw.writelines(alphabet + '\n')
                s = seq_record.seq
                
                k = 1
                for aa in s:
                    line=str(k) + ' ' + aa + ' '
                    k += 1
                    idx = alphabet.find(aa)
                    col = 0
                    for a in alphabet:
                        line = line + str( blosumMatrix[idx][col]) + ' '
                        col += 1
                    line = line + '\n'
                    pw.writelines(line)

# save each PSSM file as CSV file format. Each element is string          
def savePSSMFile2CSV(pssmfilesdir, csvfilesdir):
    listfile = os.listdir(pssmfilesdir)
    for eachfile in listfile:
        filename = eachfile.split('.')
        pssm=[]
        
        # read PSSM from ascii_pssm file
        with open(pssmfilesdir + '/' + eachfile, 'r') as pf:
            count = 0
            for eachline in pf:
                count += 1
                if count <=3:
                    continue
                if not len(eachline.strip()):
                    break
                line = eachline.split()
                pssm.append(line[2:22])
                
        # write PSSM to csv file
        with open(csvfilesdir + '/' + filename[0] + '.csv', 'w') as csvfile:
            cfw = csv.writer( csvfile)
            cfw.writerows(pssm)

# read numeric matrix from csv file            
def readPSSMFromCSVFile(filename):
    pssm=[]
    with open( filename, 'r') as csvfile:
        cfr = csv.reader(csvfile)
        for row in cfr:
            r = []
            for m in row:
                r.append(eval(m))
            pssm.append(r)
    return pssm

#  get a dict pssm   
def getPSSMMatFileFromFastafile( dirname, fastafile, matfilename, dbName='swissprot'):
    # generate the PSSM file of each protein in fastafile by psi-blast
    getPSSMFiles(fastafile,dbName)
    
    # save each PSSM file as CSV file format. Each element is string
    savePSSMFile2CSV(dirname, dirname)
    
    # geerate PSSM 
    pssm = {}    
    listf = os.listdir(dirname)
    for file in listf:
        #If file is csv format
        filename = file.split('.')
        if 'csv' in file:
            p=readPSSMFromCSVFile(dirname + '/' + file)
            pssm[filename[0]] = p
    
                               
    # save to mat file
    sio.savemat(matfilename, pssm)

# read pssm file, return numpy.array
def readPSSMFile(filename):
    pssm = []
    with open(filename, 'r') as fr:
        ls = fr.readlines()
        for ln in ls[3:]:
            t = []
            ln = ln.strip()
            if not len(ln):
                break
            strval = ln.split()
            
            for i in range(2, 22):
                t.append(1/(1 + math.exp(-eval(strval[i]))))
            
            pssm.append(t)
    return np.array(pssm)

# 读filenmae文件中的PSSM矩阵，序列最大长度maxlen，如果序列长度小于maxlen,则填充-10
# 以-2.95作为序列开始标志（原因：1/(1+exp(2.95) ~ 0.05
# 以11为序列结束标志
# 由于填充了【开始】，【结束】标志，其实最多只从PSSM序列中读取maxlen-2行
def create_padding_pssm_mask(filename, padding_position="post", maxlen=1000):
    pssm = []
    startln =  np.ones((20,)) * (2.95)
    endln = np.ones((20,))  * (-11)
    paddln = np.ones((20,)) * (10)
    
    # 把PSSM矩阵读到pssm列表
    with open(filename, 'r') as fr:
        ls = fr.readlines()
        for ln in ls[3:]:
            t = []
            ln = ln.strip()
            if not len(ln):
                break
            strval = ln.split()
            
            for i in range(2, 22):
                t.append(-eval(strval[i]))
            
            pssm.append(t)
    
    # 截取maxlen行，或填充到maxlen行
    padding_pssm = []
    mask = []
    if len(pssm) >= maxlen-2: # 不要填充
        padding_pssm.append(startln)      
        padding_pssm += pssm[:maxlen-2]
        padding_pssm.append(endln)
        mask = [0 for _ in range(maxlen)]
    else: # 需要填充
        # 计算要填充的行数
        n = maxlen - len(pssm) - 2
        if padding_position == "post":
            padding_pssm.append(startln)
            padding_pssm = padding_pssm + pssm
            padding_pssm.append(endln)
            for _ in range(n):
                padding_pssm.append(paddln)
            mask = [0 for _ in range(len(pssm) + 2)] + [1 for _ in range(n)]
        elif padding_position == "pre":
            for _ in range(n):
                padding_pssm.append(paddln)
            padding_pssm.append(startln)
            padding_pssm = padding_pssm + pssm
            padding_pssm.append(endln)
            mask = [1 for _ in range(n)] + [0 for _ in range(len(pssm) + 2)]
    return 1/(1 + np.exp(padding_pssm)), np.array(mask)
        
            
"""
fastafile = '/home/weizhong/Repoes/PDNA_CNN/PDNA_Data/TargetDNA/PDNA-543_sequence.fasta'
outdir = '/home/weizhong/Repoes/PDNA_CNN/PDNA_Data/PDNA543_PSSM/'
getPSSMFiles(fastafile, outdir)
"""
#x,mask=create_padding_pssm_mask("E:/Repoes/Enzyme/pssm/test/P0DKX6.txt", maxlen=700)


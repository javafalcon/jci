# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 10:23:53 2019

@author: falcon1
"""
'''
fpw = open('e:\\Repoes\\newSubLoc.fasta','a')

fpr = open('e:\\Repoes\\subcellLoc.txt', 'r')

for line in fpr:
    
    if line.strip() == "":
        continue
    
    else:
        fpw.write(line)
fpr.close()
fpw.close()
'''
from Bio import SeqIO
i = 1
for seq_record in SeqIO.parse('e:\\Repoes\\newSubLoc.fasta', 'fasta'):

        i = i + 1
print(i)

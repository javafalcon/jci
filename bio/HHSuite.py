#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 14:44:30 2020

@author: weizhong
"""

from Bio import SeqIO
import os
from subprocess import run

def getHHSuiteFiles(fastafile, outdir="."):
    inputfile = 'input.fasta'
    for seq_record in SeqIO.parse(fastafile, 'fasta'):
        print("\r{}".format(seq_record.id))
        if os.path.exists(inputfile):
            os.remove(inputfile)
        SeqIO.write(seq_record, inputfile, 'fasta')
        
        hitresult = os.path.join(outdir, seq_record.id)
        if os.path.exists(hitresult):
            os.remove(hitresult)
            
        cmd = ["hhblits", "-i", inputfile, "-o", hitresult, "-n", "1", "-d", "/home/weizhong/software/hh-suite/uniclust30_2018_08/uniclust30_2018_08"]
        cmd = " ".join(cmd)
        
        run(cmd, shell=True)
if __name__ == "__main__":
    fastafile = '/home/weizhong/Repoes/PDNA_CNN/PDNA_Data/TargetDNA/PDNA-543_sequence.fasta'
    outdir = '/home/weizhong/Repoes/PDNA_CNN/PDNA_Data/PDNA543_hhsuite'
    getHHSuiteFiles(fastafile,outdir)
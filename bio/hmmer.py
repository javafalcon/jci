# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 20:05:43 2019

@author: falcon1
"""
import os
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.Alphabet import IUPAC
from Bio.SeqRecord import SeqRecord

def getHomoProteinsByHMMER(seqdata, head='nohead'):
    # HMMER command line
    hmmbuildCMD = r'hmmbuild input.hmm input.fasta' 
    hmmsearchCMD = r'hmmsearch input.hmm e:/uniprot_sprot.fasta > input.out'
    if os.path.exists('input.hmm'):
        os.remove('input.hmm')
    if os.path.exists('input.fasta'):
        os.remove('input.fasta')
        
    fpw = open('input.fasta','w')
    seq = Seq(seqdata, IUPAC.ExtendedIUPACProtein)
    records = SeqRecord(seq, id=head)
    SeqIO.write(records, fpw, 'fasta')
    fpw.close()
    
    os.system(hmmbuildCMD)
    os.system(hmmsearchCMD)
    
    homoProtein=[]
    fpr = open('input.out','r')
    lines = fpr.readlines()
    fpr.close()
    
    i = 14
    while i <= 24:
        line = lines[i]
        if line.strip() == "":
            break
        if '----' in line:
            break
        s = line.split()
        homoProtein.append(s[8])
        i = i + 1

    return homoProtein


def main():
    seqdata ='SLFEQLGGQAAVQAVTAQFYANIQADATVATFFNGIDMPNQTNKTAAFLCAALGGPNAWTGRNLKEVHAN\
MGVSNAQFTTVIGHLRSALTGAGVAAALVEQTVAVAETVRGDVVTV'
    head = 'd1d1wa'
    homoProtein = getHomoProteinsByHMMER(seqdata,head)
    print(homoProtein)
    
if __name__ == '__main__':
    main()
    
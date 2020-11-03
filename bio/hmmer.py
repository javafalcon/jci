# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 20:05:43 2019

@author: falcon1
"""
import os
from subprocess import run
import scipy.io as sio
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.Alphabet import IUPAC
from Bio.SeqRecord import SeqRecord

# get a protein's homology proteins by HMMER
def getHomoProteinsByHMMER(seqrecord):
    # HMMER command line
    hmmbuildCMD = r'hmmbuild input.hmm input.fasta' 
    hmmsearchCMD = r'hmmsearch input.hmm uniprot_sprot.fasta > input.out'
    if os.path.exists('input.hmm'):
        os.remove('input.hmm')
    if os.path.exists('input.fasta'):
        os.remove('input.fasta')
    if os.path.exists('input.out'):
        os.remove('input.out')
        
    fpw = open('input.fasta','w') 
    SeqIO.write(seqrecord, fpw, 'fasta')
    fpw.close()
    
    #os.system(hmmbuildCMD)
    #os.system(hmmsearchCMD)
    run(hmmbuildCMD, shell=True)
    run(hmmsearchCMD, shell=True)
    
    homoProtein=[]
    fpr = open('input.out','r')
    lines = fpr.readlines()
    fpr.close()
    
    if not lines:
        return []
    
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

def buildFunctionDomainSet(dataFile, seqs_dict=False):
    if seqs_dict:
        uniprot_seqs_dict = sio.loadmat('uniprot_seqs_dict.mat')
    else:
        uniprot_seqs_dict = {}
        for uniprot_seq_record in SeqIO.parse('uniprot_sprot.fasta', 'fasta'):
            uniprot_seqs_dict[uniprot_seq_record.id] = str(uniprot_seq_record.seq)
        
    
    pfams = {}
    hmmscanCMD = 'hmmscan -o out.txt --tblout fmout.tbl --acc --noali Pfam-A.hmm input.fasta'
    num_total = 1
    num_nohom = 0
    for seq_record in SeqIO.parse(dataFile, 'fasta'):
        print("{}: {}".format(num_total,seq_record.id))
        num_total += 1
        
        pfam = []
        h = getHomoProteinsByHMMER(seq_record)
        if len(h) == 0:
            num_nohom += 1
            print("{}/{} has not homology proteins".format(seq_record.id, num_nohom))
        else:
            for j in range(len(h)):
                pid = h[j]
                pseq = Seq(uniprot_seqs_dict[pid],IUPAC.ExtendedIUPACProtein)
                if os.path.exists('out.txt'):
                    os.remove('out.txt') 
                if os.path.exists('fmout.tbl'):
                    os.remove('fmout.tbl')
                if os.path.exists('input.fasta'):
                    os.remove('input.fasta')
                    
                fpw = open('input.fasta','w') 
                seqrecord = SeqRecord(pseq, id=pid)
                SeqIO.write(seqrecord, fpw, 'fasta')
                fpw.close()
                
                run(hmmscanCMD, shell=True)
                
                with open('fmout.tbl','r',encoding='UTF-8') as fm:
                    flag = 1
                    lines = fm.readlines()
                    for tline in lines:
                        if '[ok]' in tline:
                            break
                        elif tline.startswith('#'):
                            if flag == 1:
                                continue
                            else:
                                break
                        else:
                            flag = 0
                            s = tline.split()
                            pf = s[1][:7]
                            pfam.append(pf)
                    # end with
               # end for j in range(len(h))
        pfams[seq_record.id] = pfam 
    # end for seq_record 
    return pfams

# example of usage
if __name__ == "__main__":
    """
    seqdata ="".joint(['SLFEQLGGQAAVQAVTAQFYANIQADATVATFFNGIDMPNQTNKTAAFLCAALGGPNAWTGRNLKEVHAN', 
                       'MGVSNAQFTTVIGHLRSALTGAGVAAALVEQTVAVAETVRGDVVTV'])
    head = 'd1d1wa'
    seq = Seq(seqdata, IUPAC.ExtendedIUPACProtein)
    seqrecord = SeqRecord(seq, id=head)
    homoProtein = getHomoProteinsByHMMER(seqrecord)
    print(homoProtein)
    """
    
    pfams = buildFunctionDomainSet(r'/home/weizhong/Repoes/RemoteHomology/program/3187seqs.fasta')
    #sio.savemat('independent_pfams.mat',pfams)
    import pickle as pk
    with open('3187_pfams.data', 'wb') as f:
        pk.dump(pfams, f)
    
    
        
    
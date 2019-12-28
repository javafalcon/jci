# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 10:35:47 2019

@author: Administrator
"""

from Bio import SeqIO
def uniprotSeqs():
    uniprot_seqs_dict = {}
    for seq_record in SeqIO.parse('uniprot_sprot.fasta', 'fasta'):
        uniprot_seqs_dict[seq_record.id] = str(seq_record.seq)
    return uniprot_seqs_dict


import random
def aa2code(aa:str):
    code={}
    code['A'] = ['GCU','GCC','GCA','GCG']
    code['C'] = ['UGU','UGC']
    code['D'] = ['GAU','GAC']
    code['E'] = ['GAA','GAG']
    code['F'] = ['UUU','UUC']
    code['H'] = ['GAU','GAC']
    code['I'] = ['AUU','AUC','AUA']
    code['G'] = ['GGU','GGC','GGA','GGG']
    code['K'] = ['AAA','AAG']
    code['L'] = ['UUA','UUG','GUU','GUC','GUA','GUG']
    code['M'] = ['AUG']
    code['N'] = ['AAU','AAC']
    code['Q'] = ['CAA','CAG']
    code['P'] = ['CCU','CCC','CCA','CCG']
    code['R'] = ['CGU','CGC','CGA','CGG','AGA','AGG']
    code['S'] = ['UCU','UCC','UCA','UCG']
    code['T'] = ['ACU','ACC','ACA','ACG']
    code['V'] = ['GUU','GUC','GUA','GUG']
    code['W'] = ['UGG']
    code['Y'] = ['UAU','UAC']
    code['#'] = ['XXX']
    c = code[aa]
    return random.choice(c)
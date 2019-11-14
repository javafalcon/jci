# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 10:35:47 2019

@author: Administrator
"""

from Bio import SeqIO

uniprot_seqs_dict = {}
for seq_record in SeqIO.parse('uniprot_sprot.fasta', 'fasta'):
    uniprot_seqs_dict[seq_record.id] = str(seq_record.seq)
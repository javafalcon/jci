# -*- coding: utf-8 -*-
"""
关于Gene Ontology的使用
对一个蛋白质序列，先使用HMMER软件从swiss-prot中找出同源（前10）蛋白质
对每一个同源蛋白质在GO数据库中查找GO的注释功能编号
把所有同源蛋白质的GO注释合成一个集合，集合中每个元素是形如GO:xxxxxxx的注释

@author: falcon1
"""
from hmmer import getHomoProteinsByHMMER
from Bio.Seq import Seq
from Bio.Alphabet import IUPAC
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import json
import pymysql 

SELECT_TERM_ACC = "SELECT  distinct term.acc  \
                   FROM   gene_product \
                          INNER JOIN dbxref ON (gene_product.dbxref_id=dbxref.id) \
                          INNER JOIN species ON (gene_product.species_id=species.id) \
                          INNER JOIN association ON (gene_product.id=association.gene_product_id) \
                          INNER JOIN evidence ON (association.id=evidence.association_id) \
                          INNER JOIN term ON (association.term_id=term.id) \
                   WHERE  term.is_obsolete=0  and   dbxref.xref_key =%s  ORDER BY term.acc;"
                   
def queryTerms(sql, args=None):
    connection = pymysql.connect(
            host = 'localhost',
            user = 'root',
            passwd = 'jciicpr',
            db = 'go20160604'
    ) #连接数据库
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql, args)
            result = cursor.fetchall() #返回结果（元祖）
            
    finally:
        connection.close()
    
    return result

def getGOSet(seqdata):
    # get seq's homology proteins list
    seq = Seq(seqdata, IUPAC.ExtendedIUPACProtein)
    seqrecord = SeqRecord(seq, id='None')
    homoProtein = getHomoProteinsByHMMER(seqrecord)
    goset = set()
    
    for p in homoProtein:
        plist = p.split('|')
        pid = plist[1]
        result = queryTerms(SELECT_TERM_ACC, pid)
        goset.update(result)
        #print(len(goset))
    
    return goset

def main():
    i = 1
    prot_go_set_dict = {}
    for seq_record in SeqIO.parse('e:\\Repoes\\SubLoc.fasta', 'fasta'):
        goes = getGOSet(str(seq_record.seq))
        prot_go_set_dict[seq_record.id] = goes
        print(i)
        i = i + 1
    
    with open('e:\\Repoes\\SubLocGOExp.json','a') as fw:
       json.dump(prot_go_set_dict, fw,ensure_ascii=False)
       fw.write('\n')

if __name__ == '__main__':
    main()
    
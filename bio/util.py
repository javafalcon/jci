# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 10:28:24 2019

@author: Administrator
"""
class PfamClan(object):
    def __init__(self,cid,name,abb,annot):
        self.cid = cid
        self.name = name
        self.abb = abb
        self.annot = annot

"""
compulate distance between pfam domain pf1 and pf2
"""
def pfamcmp(pf1,pf2,clanDict):
    s = 0
    pfset1 = set(pf1)
    pfset2 = set(pf2)
    if not pfset1 or not pfset2:
        return s
    
    
    intset = pfset1.intersection(pfset2)
    uniset = pfset1.union(pfset2)
    
    if not intset:
        for p1 in pfset1:
            clan1=clanDict[p1]
            for p2 in pfset2:
                clan2 = clanDict[p2]
                if not clan1.cid or not clan2.cid:
                    s = 0
                else:
                    if clan1==clan2:
                        s=0.2
    else:
        s = len(intset)/len(uniset)
    return s

"""
从psi-blast运行得到的PSSM文件中读入PSSM矩阵
返回一个二维的列表
"""
def readPSSMFromFile(filename):
    pssm = []
    with open(filename, 'r') as fo:
        lines = fo.readlines()
        lines = lines[3:]
        for line in lines:
            if len(line) < 2:
                break
            else:
                s = line.split(" ")
                k = 0
                m = []
                for e in s:
                    if len(e) > 0:
                        k += 1
                        if 2<k<23:
                            m.append(eval(e))
                pssm.append(m)
            
    return pssm

def main():
    filename=r'E:\Repoes\jcilwz\RemoteHomology\program\scope_independent_PSSM\d1a7sa_.txt';
    pssm = readPSSMFromFile(filename)
    print(pssm)

if __name__ == "__main__":
    main()
                    
        


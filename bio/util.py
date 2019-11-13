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

def pfamcmp(pf1,pf2,clanDict):
    s = 0
    if not pf1 or not pf2:
        return s
    pfset1 = set(pf1)
    pfset2 = set(pf2)
    
    intset = pfset1.intersection(pfset2)
    uniset = pfset1.union(pfset2)
    
    if not intset:
        for p1 in pf1:
            clan1=clanDict[p1].cid
            for p2 in pf2:
                clan2 = clanDict[p2].cid
                if not clan1 or not clan2:
                    s = 0
                else:
                    if clan1==clan2:
                        s=0.2
    else:
        s = len(intset)/len(uniset)
    return s

#import scipy.io as sio
#data=sio.loadmat(r'E:\Repoes\jcilwz\RemoteHomology\program\independent_pfams.mat')
#clanDict=sio.loadmat(r'E:\Repoes\jcilwz\RemoteHomology\program\pfamClan.mat')
s=pfamcmp(list(pf1),list(pf2),clanDict)
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 11:00:38 2019

@author: Administrator
"""
import numpy as np
import math

"""
计算GM(1,1)模型参数
x是序列
"""
from numpy.linalg import inv
def GMParam(x,model=1):
    n = len(x)
    X = np.zeros(n,)
    X[0] = x[0]
    for i in range(1,n):
        X[i] = X[i-1] + x[i]
    
    Z = np.zeros(n,)
    for i in range(1,n):
        Z[i] = 0.5 * (X[i] + X[i-1])
    
    if model == 1:
        B = np.zeros((n-1,2))
        for i in range(n-1):
            B[i,0] = -Z[i+1]
            B[i,1] = 1
            
        Y = np.zeros(n-1,)
        for i in range(n-1):
            Y[i] = x[i+1]
    if model == 2:
        B = np.zeros((n-1,3))
        for i in range(n-1):
            B[i,0] = -X[i+1]
            B[i,1] = -Z[i+1]
            B[i,2] = 1        
    
        Y = np.zeros(n-1,)
        for i in range(n-1):
            Y[i] = X[i+1] - X[i]
    
    TB = np.transpose(B)
    a = np.matmul(TB, B)
    a = inv(a)
    a = np.matmul(a,TB)
    a = np.matmul(a,Y)
        
    return a

"""
计算参考序列y与比较序列集X中每个序列的灰色关联度
"""
def GreyIncidenceDegree(y,X,p=0.5):
    m = len(X)
    n = len(y)
    D = np.zeros(shape=(m,n))
    z = np.zeros(shape=(m,n))
    R = np.zeros(shape=(m,))
    
    #计算关联度
    for i in range(m):
        D[i,:] = abs(y - X[i,:])
    dmin = np.min(D)
    dmax = np.max(D)
    
    for i in range(m):
        for j in range(n):
            z[i,j] = (dmin + p * dmax) / (D[i,j] + p * dmax)
            
    R = np.sum(z, axis=1) / n
    
    return R

"""
计算由pssm矩阵生成Pse-PSSM AAC向量。
pssm是蛋白质的进化信息特异位置打分矩阵,L-by-20
n=1，使用GM(1,1)模型产生伪成分
n=2，使用GM(2,1)模型产生伪成分
首先对pssm矩阵以函数1/(1+exp(-x))变换，使得矩阵每个元素都是（0，1）之间的正数
"""
def greyPsePSSM(pssm, n=1):
    sp = pssm.shape
    # 以1/(1+exp(-x))变换pssm矩阵
    for i in range(sp[0]):
        for j in range(sp[1]):
            pssm[i,j] = 1 / (1 + math.exp(-pssm[i,j]))
            
    if n == 1: # generate pseudo composition by GM(1,1)
        psem = np.ndarray((60,))
        psem[:20] = np.mean(pssm,axis=0)
        
        
def main():
    # 灰色关联度
    """
    X = [[2045.3, 1942.2, 1637.2, 1884.2, 1602.3],
         [34374, 31793, 27319, 32516, 16297],
         [14.6792, 14.8449, 1.4774, 46.604, 9.4959],
         [120.9, 100.1, 65.9, 80.52, 54.22],
         [0.3069, 0.7409, 0.361, 3.7, 2.0213],
         [49.4201, 34.8699, 50.974, 50.4325, 40.8828]]
    X = np.array(X)
    X1 = np.ndarray(shape=X.shape)
    for i in range(6):
        for j in range(5):
            X1[i,j] = X[i,j]/X[i,0]
    
    y = [1,0.9496,0.8005,0.9212,0.7834]
    y = np.array(y)
    R = GreyIncidenceDegree(y,X1,p=0.5)
    print(R)
    """
    
    # 灰色模型
    x = [2.874,3.278,3.337,3.390,3.679]
    a = GMParam(x,2)
    print(a)
if __name__ == "__main__":
    main()
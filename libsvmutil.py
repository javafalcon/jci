# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 12:49:31 2018

@author: Administrator
"""

from svmutil import svm_train
# 使用Grid Search方法寻找训练集（X，y）使用rbf核函数的最优参数
def libsvm_grid_search(X,y):
    best_c, best_g, best_acc = 0, 0, 0
    for c in [pow(2,i) for i in range(-5,17,2)]:
        for g in [pow(2,j) for j in range(-15,5,2)]:
            param = ['-c', str(c), '-g', str(g), '-v', '5']
            acc = svm_train(y,X, " ".join(param))
            if acc > best_acc:
                best_c, best_g, best_acc = c, g, acc
    
    return best_c, best_g, best_acc


            
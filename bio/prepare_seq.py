# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 09:36:27 2020

@author: lwzjc
"""
import re
from tensorflow import keras
def protseq_to_vec(seqs, padding_position="post",  maxlen=1000):
    """
    把由氨基酸字母表示的蛋白质序列转化为字向量.
    字表共包括24个字符：21个字母（20个氨基酸字母+'X'）,开始标识，结束标识，和填充的标识.
    Parameters
    ----------
    seqs : list
        蛋白质序列列表. 如：['MKF...', 'MLA..', ...]
    padding_position : str, optional
        序列填充方式，如果是None则不填充. The default is "post".
    maxlen : , optional
        截取蛋白质序列的最大长度. The default is 1000.

    Returns
    -------
    X : numpy.array.
    """
    amino_acids = "ARNDCQEGHILKMFPSTWYVX"
    regexp = re.compile('[^ARNDCQEGHILKMFPSTWYVX]')
    X = []
    for i in range(len(seqs)):
        seq = seqs[i]
        seq = regexp.sub('X', seq)
        # 把蛋白质序列按氨基酸转换为数字编码
        t = [22] # 22 是序列开始标识
        for a in seq:
            t.append(amino_acids.index(a)+1)
        t.append(23) # 23是序列结束表示    
        X.append(t)
    if padding_position is not None:
        X = keras.preprocessing.sequence.pad_sequences(X, maxlen=maxlen,
                                                         padding=padding_position,
                                                         truncating=padding_position)   
    return X
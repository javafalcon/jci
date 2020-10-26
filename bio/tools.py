# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 10:35:47 2019

@author: Administrator
"""

from Bio import SeqIO
import numpy as np

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

def plot_history(history):
    import matplotlib.pyplot as plt
    #%matplotlib inline
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(1, len(acc) + 1)
    
    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.show()
    
    plt.clf()   # clear figure    
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.show()
    
def displayMetrics(y_true, y_score, threshold=0.5):
    from sklearn import metrics
    y_pred = (y_score > threshold).astype(float)
    cm = metrics.confusion_matrix(y_true, y_pred)
    print("confusion_matrix:\n", cm)
    acc = metrics.accuracy_score(y_true, y_pred)
    print("accuracy:", acc)
    mcc = metrics.matthews_corrcoef(y_true, y_pred)
    print("MCC:", mcc)
    auc = metrics.roc_auc_score(y_true, y_score)
    print("AUC:", auc)

def displayMLMetrics(y_true, y_pred, fileName, info):
    # output Multi-lable classifier's Metrics
    from sklearn import metrics
    with open(fileName, 'a') as fw:
        fw.write(info)
        fw.write("hamming loss = {}\n".format(metrics.hamming_loss(y_true, y_pred)))
        fw.write("subset accuracy = {}\n".format( metrics.accuracy_score(y_true, y_pred)))
        fw.write("macro average precision_score: {}\n".format(metrics.average_precision_score(y_true,y_pred,average="macro")))
        fw.write("micro average precisioin_score: {}\n".format(metrics.average_precision_score(y_true,y_pred,average="micro")))
    
def plot_cm(labels, predictions, p=0.5):
    from sklearn.metrics import confusion_matrix
    #import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns
    #mpl.rcParams['figure.figsize'] = (12, 10)
    #colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    cm = confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    
    print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
    print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
    print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
    print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
    print('Total Fraudulent Transactions: ', np.sum(cm[1]))    
      
      
    
    
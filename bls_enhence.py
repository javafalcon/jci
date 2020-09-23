import numpy as np
from sklearn import preprocessing
from numpy import random
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd

def show_accuracy(predictLabel,Label):
    Label = np.ravel(Label).tolist()
    predictLabel = predictLabel.tolist()
    count = 0
    for i in range(len(Label)):
        if Label[i] == predictLabel[i]:
            count += 1
    return (round(count/len(Label),5))

class scaler:
    def __init__(self):
        self._mean = 0
        self._std = 0
    
    def fit_transform(self,traindata):
        self._mean = traindata.mean(axis = 0)
        self._std = traindata.std(axis = 0)
        return (traindata-self._mean)/self._std
    
    def transform(self,testdata):
        return (testdata-self._mean)/self._std

class node_generator:
    def __init__(self,whiten = False):
        self.Wlist = []
        self.blist = []
        self.nonlinear = 0
        self.whiten = whiten
    
    def sigmoid(self,data):
        return 1.0/(1+np.exp(-data))
    
    def linear(self,data):
        return data
    
    def tanh(self,data):
        return (np.exp(data)-np.exp(-data))/(np.exp(data)+np.exp(-data))
    
    def relu(self,data):
        return np.maximum(data,0)
    
    def orth(self,W):
        for i in range(0,W.shape[1]):
            w = np.mat(W[:,i].copy()).T
            w_sum = 0
            for j in range(i):
                wj = np.mat(W[:,j].copy()).T
                w_sum += (w.T.dot(wj))[0,0]*wj 
            w -= w_sum
            w = w/np.sqrt(w.T.dot(w))
            W[:,i] = np.ravel(w)
        return W
        
    def generator(self,shape,times):
        for i in range(times):
            W = 2*random.random(size=shape)-1
            if self.whiten == True:
                W = self.orth(W)
            b = 2*random.random()-1
            yield (W,b)
    
    def generator_nodes(self, data, times, batchsize, nonlinear):
        self.Wlist = [elem[0] for elem in self.generator((data.shape[1],batchsize),times)]
        self.blist = [elem[1] for elem in self.generator((data.shape[1],batchsize),times)]
        
        self.nonlinear = {'linear':self.linear,
                          'sigmoid':self.sigmoid,
                          'tanh':self.tanh,
                          'relu':self.relu
                          }[nonlinear]
        nodes = self.nonlinear(data.dot(self.Wlist[0])+self.blist[0])
        for i in range(1,len(self.Wlist)):
            nodes = np.column_stack((nodes, self.nonlinear(data.dot(self.Wlist[i])+self.blist[i])))
        return nodes
        
    def transform(self,testdata):
        testnodes = self.nonlinear(testdata.dot(self.Wlist[0])+self.blist[0])
        for i in range(1,len(self.Wlist)):
            testnodes = np.column_stack((testnodes, self.nonlinear(testdata.dot(self.Wlist[i])+self.blist[i])))
        return testnodes   

    def update(self,otherW, otherb):
        self.Wlist += otherW
        self.blist += otherb


class broadnet_enhence:
    def __init__(self, 
                 maptimes = 10, 
                 enhencetimes = 10,
                 traintimes = 100,
                 map_function = 'linear',
                 enhence_function = 'linear',
                 batchsize = 'auto', 
                 acc = 1,
                 step = 1,
                 reg = 0.001):
        
        self._maptimes = maptimes
        self._enhencetimes = enhencetimes
        self._batchsize = batchsize
        self._traintimes = traintimes
        self._acc = acc
        self._step = step
        self._reg = reg
        self._map_function = map_function
        self._enhence_function = enhence_function
        
        self.W = 0
        self.pesuedoinverse = 0
        
        self.normalscaler = scaler()
        self.onehotencoder = preprocessing.OneHotEncoder(sparse = False)
        self.mapping_generator = node_generator()
        self.enhence_generator = node_generator(whiten = True)

    def fit(self,oridata,orilabel):
        if self._batchsize == 'auto':
            self._batchsize = oridata.shape[1]
        data = self.normalscaler.fit_transform(oridata)
        label = self.onehotencoder.fit_transform(np.mat(orilabel).T)
        
        mappingdata = self.mapping_generator.generator_nodes(data,self._maptimes,self._batchsize,self._map_function)
        enhencedata = self.enhence_generator.generator_nodes(mappingdata,self._enhencetimes,self._batchsize,self._enhence_function)
        inputdata = np.column_stack((mappingdata,enhencedata))
        
        self.pesuedoinverse = self.pinv(inputdata)
        self.W =  self.pesuedoinverse.dot(label)
        
        Y = self.predict(oridata)
        accuracy, i = self.accuracy(Y,orilabel),0
        print("inital setting, number of mapping nodes {0}, number of enhence nodes {1}, accuracy {2}".format(mappingdata.shape[1],enhencedata.shape[1],round(accuracy,5)))
        
        while i < self._traintimes and accuracy < self._acc:
            Y = self.addingenhence_predict(data, label, self._step,self._batchsize)  
            accuracy = self.accuracy(Y,orilabel)
            i += 1
            print("enhencing {3}, number of mapping nodes {0}, number of enhence nodes {1}, accuracy {2}".format(len(self.mapping_generator.Wlist)*self._batchsize,len(self.enhence_generator.Wlist)*self._batchsize,round(accuracy,5),i))

    def pinv(self,A):
        return np.mat(self._reg*np.eye(A.shape[1])+A.T.dot(A)).I.dot(A.T)
    
    def decode(self,Y_onehot):
        Y = []
        for i in range(Y_onehot.shape[0]):
            lis = np.ravel(Y_onehot[i,:]).tolist()
            Y.append(lis.index(max(lis)))
        return np.array(Y)
    
    def accuracy(self,predictlabel,label):
        label = np.ravel(label).tolist()
        predictlabel = predictlabel.tolist()
        count = 0
        for i in range(len(label)):
            if label[i] == predictlabel[i]:
                count += 1
        return (round(count/len(label),5))
        
    def predict(self,testdata):
        testdata = self.normalscaler.transform(testdata)
        test_inputdata = self.transform(testdata)   
        return self.decode(test_inputdata.dot(self.W))   
    
    def transform(self,data):
        mappingdata = self.mapping_generator.transform(data)
        enhencedata = self.enhence_generator.transform(mappingdata)
        return np.column_stack((mappingdata,enhencedata))
    
    def addingenhence_nodes(self, data, label, step = 1, batchsize = 'auto'):
        if batchsize == 'auto':
            batchsize = data.shape[1]
            
        mappingdata = self.mapping_generator.transform(data)
        inputdata = self.transform(data)         
        localenhence_generator = node_generator()
        extraenhence_nodes = localenhence_generator.generator_nodes(mappingdata,step,batchsize,self._enhence_function)
            
        D = self.pesuedoinverse.dot(extraenhence_nodes)
        C = extraenhence_nodes - inputdata.dot(D)
        BT = self.pinv(C) if (C == 0).any() else  np.mat((D.T.dot(D)+np.eye(D.shape[1]))).I.dot(D.T).dot(self.pesuedoinverse)
        
        self.W = np.row_stack((self.W-D.dot(BT).dot(label),BT.dot(label))) 
        self.enhence_generator.update(localenhence_generator.Wlist,localenhence_generator.blist)
        self.pesuedoinverse =  np.row_stack((self.pesuedoinverse - D.dot(BT),BT)) 
    
    def addingenhence_predict(self, data, label, step = 1, batchsize = 'auto'):
        self.addingenhence_nodes(data, label, step, batchsize)
        test_inputdata = self.transform(data) 
        return self.decode(test_inputdata.dot(self.W)) 
    

iris=load_iris()
X = iris.data
Y = iris.target
traindata,testdata,trainlabel,testlabel = train_test_split(X,Y,test_size=0.2,random_state = 2018)
#print(traindata.shape,trainlabel.shape,testdata.shape,testlabel.shape)

#data = pd.read_excel('/Users/zhuxiaoxiansheng/Desktop/日常/数据集/糖尿病数据集.xlsx')
#label = np.mat(data['标签'].values)
#data = data.drop('标签',axis = 1)
#traindata,testdata,trainlabel,testlabel = train_test_split(data.values,np.ravel(label),test_size=0.2,random_state = 2018)

bls = broadnet_enhence(maptimes = 10, 
                       enhencetimes = 10,
                       traintimes = 10,
                       map_function = 'tanh',
                       enhence_function = 'sigmoid',
                       batchsize = 1, 
                       acc = 1,
                       step = 5,
                       reg = 0.001)

bls.fit(traindata,trainlabel)
predictlabel = bls.predict(testdata)
print(show_accuracy(predictlabel,testlabel))


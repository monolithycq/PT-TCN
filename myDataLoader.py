import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader


class myMultiDataSet(Dataset):
    '''
    读数据，加滑动窗口
    '''
    def __init__(self,start,end,sw_width,n_out,f1_address = 'data\粗选槽特征数据.xls'):
        '''
        :param sw_width: 滑动窗口长度
        :param n_out: 预测未来长度
        :param f1_address: 文件名
        '''
        self.sw_width = sw_width
        self.n_out = n_out
        df = pd.read_excel(f1_address,header=1,index_col=0)
        initialData = df.values
        initDataSet = initialData[start:end,:]
        self.initialdatalen, self.var_num = initDataSet.shape[0],initDataSet.shape[1]-1
        self.samples_num = self.initialdatalen-self.sw_width-self.n_out+1
        self.sample, self.label = self.__sliding_windows(initDataSet)
        print('sample', self.sample.shape)

    def __sliding_windows(self,data):#将原始数据通过滑动窗口改为序列数据
        X = torch.zeros((self.samples_num,self.sw_width,self.var_num+1))
        y = torch.zeros((self.samples_num,1,self.var_num+1))
        for i in range(self.samples_num-1):
            start ,end = i, i + self.sw_width
            X[i,:,:] = torch.from_numpy(data[start:end,:])
            y[i,:,:] = torch.from_numpy(data[end:end+1,:])
        return (X,y)

    def __len__(self):
        return self.samples_num

    def __getitem__(self, idx):
        return self.sample[idx,:,:],self.label[idx,:,:]

class  myAutoDataSet_v1(Dataset):
    def __init__(self,sw_width,n_out,covariates_idx,y_idx,initDataSet):
        '''

        :param start:
        :param end:
        :param sw_width: 滑动窗口长度
        :param n_out: 预测未来长度
        :param covariates_idx: 协变量idx
        :param y_idx:
        :param f1_address: 文件名
        :param header: 文件中header
        :param index_col: 文件中index-col
        '''
        self.sw_width = sw_width
        self.n_out = n_out
        self.initialdatalen, self.var_num = initDataSet.shape[0], len(covariates_idx)
        self.samples_num = self.initialdatalen - self.sw_width - self.n_out
        self.sample, self.label = self.__sliding_windows(initDataSet,covariates_idx,y_idx)

    def __sliding_windows(self,data,covariates_idx,y_idx):#将原始数据通过滑动窗口改为序列数据
        covariates_len,y_len = len(covariates_idx),len(y_idx)
        X = torch.zeros((self.samples_num,self.sw_width,covariates_len+y_len))
        y = torch.zeros((self.samples_num,self.n_out,y_len))

        for i in range(self.samples_num):
            start ,end = i, i + self.sw_width
            X[i, :,:covariates_len] = torch.from_numpy(data[start:end,covariates_idx])
            X[i,:,covariates_len:covariates_len+y_len] = torch.from_numpy(data[start:end,y_idx])
            y[i,:,:] = torch.from_numpy(data[end:end+self.n_out,y_idx])
        return (X,y)
    def __len__(self):
        return self.samples_num

    def __getitem__(self, idx):
        return self.sample[idx,:,:],self.label[idx,:,:]






#(B,f,L)
class myAutoDataSet(Dataset):
    '''
    读数据，加滑动窗口
    '''
    def __init__(self,start,end,sw_width,n_out,f1_address = 'data\粗选槽特征数据.xls'):
        '''
        :param sw_width: 滑动窗口长度
        :param n_out: 预测未来长度
        :param f1_address: 文件名
        '''
        self.sw_width = sw_width
        self.n_out = n_out
        df = pd.read_excel(f1_address,header=1,index_col=0)
        initialData = df.values
        initDataSet = initialData[start:end+1,:]
        self.initialdatalen, self.var_num = initDataSet.shape[0],initDataSet.shape[1]-1
        self.samples_num = self.initialdatalen-self.sw_width-self.n_out+1
        self.sample, self.label = self.__sliding_windows(initDataSet)
        print('sample', self.sample.shape)

    def __sliding_windows(self,data):#将原始数据通过滑动窗口改为序列数据
        X = torch.zeros((self.samples_num,self.sw_width,self.var_num+1))
        y = torch.zeros((self.samples_num,1,1))
        for i in range(self.samples_num-1):
            start ,end = i, i + self.sw_width
            X[i, :, 0] = torch.from_numpy(data[start:end, -1])
            X[i,:,1:] = torch.from_numpy(data[start:end,:-1])
            y[i,:,:] = torch.from_numpy(data[end:end+1,-1])
        return (X,y)

    def __len__(self):
        return self.samples_num

    def __getitem__(self, idx):
        return self.sample[idx,:,:],self.label[idx,:,:]






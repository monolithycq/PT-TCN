import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import preprocessing
import matplotlib.pyplot as plt
import copy
from treelib import Tree, Node

def min_max_and_save(initData,file_address):
    '''

    :param initData: 初始数据 (L,K)
    :param file_address:
    :return: 最大最小以及归一化数据
    '''
    dataMax = initData.max(axis=0)
    dataMin = initData.min(axis=0)
    min_max_Data = (initData - dataMin) / (dataMax - dataMin)
    dataFile = pd.DataFrame(min_max_Data)
    writer = pd.ExcelWriter(file_address)  # 写入Excel文件
    dataFile.to_excel(writer, float_format='%.6f')
    writer.save()
    writer.close()
    return dataMax,dataMin,min_max_Data

def PearsonAnalysis(df):

    plt.figure(figsize=(12, 12))
    a = df.corr()
    sns.heatmap(df.corr(), linewidths=0.1, vmax=1.0, square=True, linecolor='white', annot=True)
    plt.show()


def crosscorr(datax, datay, lag=0, wrap=False):
    """ Lag-N cross correlation.
    Shifted data filled with NaNs

    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length
    Returns
    ----------
    crosscorr : float
    """
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else:
        return datax.corr(datay.shift(lag))

def lagPearson(df,lagBound):
    _,length =df.shape
    zeros_np = np.zeros([length,length])
    zeros_np2 = np.zeros([length,length])
    value = pd.DataFrame(zeros_np)
    offset_index = pd.DataFrame(zeros_np2)
    # deep_copy = copy.deepcopy(value)
    for i in range(length):


        for j in range(i+1):
            di = df.iloc[:,i]
            dj = df.iloc[:,j]
            rs = [crosscorr(di, dj, lag) for lag in range(-lagBound, lagBound)]

            offset = np.ceil(len(rs) / 2) - np.argmax(np.abs(rs))
            rs_abs_max = rs[np.argmax(np.abs(rs))]
            print(rs_abs_max)
            if i !=16 and j!=16:
                value.iloc[i,j] = rs_abs_max
                value.iloc[j, i] = rs_abs_max
                # stri= str(i)
                # value.iloc[value.index == i,j] = rs_abs_max
                # value.iloc[value.index == j, i] = rs_abs_max
                offset_index.iloc[i, j] = offset
                offset_index.iloc[j, i] = offset





    f = plt.figure(figsize=(12,26))
    ncols = 1
    ax = f.subplots(2, ncols)

    # ax1 = f.add_subplot(2, 1, 1)
    value_sns=sns.heatmap(value, linewidths=0.1, vmax=1.0, square=True, linecolor='white', annot=True,ax=ax[0])
    # plt.xticks([]), plt.yticks([])
    # plt.subplots_adjust(hspace=2)
    # ax2 = f.add_subplot(2, 1, 2)
    offset_sns=sns.heatmap(offset_index, linewidths=0.1, square=True,linecolor='white', annot=True,ax=ax[1])
    plt.show()


def selfCorr(df,idx):
    value = df.values
    t,k = value.shape
    # print(t,k)
    f = plt.figure(figsize=(18, 6))
    ncols = 1
    ax = f.subplots(ncols, len(idx))
    for i in range(len(idx)):
        x = value[:-1,idx[i]]
        y = value[1:,idx[i]]
        ax[i].scatter(x,y)
        ax[i].plot((0, 1), (0, 1), transform=ax[i].transAxes, ls='--', c='k', label="1:1 line")
    plt.show()



def creatTree():
    rawTree = Tree()
    rawTree.create_node(tag='0', identifier='root', data=[0, 1, 0, 0, 0]);
    a0, a00, a01, a02, a000,a001 = [4, 3, 3, 3, 3,3]
    a0, a00, a01, a02, a000,a001 = [4, 2, 2, 2, 3,3]
    #data:[kernel_size,mask_flag,dilation_bottom,level,]
    for i in range(a0):
        rawTree.create_node('0' + str(i), i, parent='root', data=[2 + i, 1, 2, 0, 16])
    for i in range(a00):
        rawTree.create_node('00' + str(i), i + a0, parent=0, data=[2 + i, 1, 3, 1, 32])
    for i in range(a01):
        rawTree.create_node('01' + str(i), i + a0 + a00, parent=1, data=[3 + i, 1, 2, 1, 32])
    for i in range(a02):
        rawTree.create_node('02' + str(i), i + a0 + a00+a01, parent=2, data=[3 + i, 1, 2, 1, 32])
    for i in range(a000):
        rawTree.create_node('000' + str(i), i + a0 + a00 + a01+a02, parent=4, data=[2 + i, 1, 2, 2, 32])
    # for i in range(a001):
    #     rawTree.create_node('001' + str(i), i + a0 + a00 + a01+a02+a001, parent=5, data=[3 + i, 1, 2, 2, 32])
    all_nodes = rawTree.all_nodes()
    node_ids = [node.identifier for node in all_nodes]
    # print(node_ids)
    return rawTree,node_ids

def creatTreeTE():
    rawTree = Tree()
    rawTree.create_node(tag='0', identifier='root', data=[0, 1, 0, 0, 0]);
    a0, a00, a01, a02, a000,a001,a0000 = [4, 3, 3, 3, 3,3,3]
    # a0, a00, a01, a02, a000,a001 = [3, 2, 2, 3, 3,3]
    #data:[kernel_size,mask_flag,dilation_bottom,level,]
    for i in range(a0):
        rawTree.create_node('0' + str(i), i, parent='root', data=[2 + i, 1, 2, 0, 64])
    for i in range(a00):
        rawTree.create_node('00' + str(i), i + a0, parent=0, data=[2 + i, 1, 3, 1, 64])
    for i in range(a01):
        rawTree.create_node('01' + str(i), i + a0 + a00, parent=1, data=[3 + i, 1, 2, 1, 64])
    for i in range(a02):
        rawTree.create_node('02' + str(i), i + a0 + a00+a01, parent=2, data=[3 + i, 1, 2, 1, 64])
    for i in range(a000):
        rawTree.create_node('000' + str(i), i + a0 + a00 + a01+a02, parent=4, data=[2 + i, 1, 2, 2, 32])
    for i in range(a001):
        rawTree.create_node('001' + str(i), i + a0 + a00 + a01+a02+a001, parent=5, data=[3 + i, 1, 2, 2, 32])
    for i in range(a0000):
        rawTree.create_node('0000' + str(i), i + a0 + a00 + a01 + a02 + a001+a0000, parent=13, data=[2 + i, 1, 2, 3, 32])
    all_nodes = rawTree.all_nodes()
    node_ids = [node.identifier for node in all_nodes]
    print(node_ids)
    return rawTree,node_ids

def father_childDict(tree):
    dic_father,dic_child ={},{}
    all_nodes = tree.all_nodes()
    for node in all_nodes:
        node_id = node.identifier
        parent_node = tree.parent(node_id)
        child_nodes = tree.children(node_id)
        if parent_node is not None:
            dic_father[node_id] = parent_node.identifier
        if child_nodes :
            dic_child[node_id] = [child_node.identifier for child_node in child_nodes]
    return dic_father,dic_child


def treeStructUpdate(tree,mask_ids):
    #更新新的树结构和fatherList，childList
    #函数中对Tree对象进行的修改会反映到函数外部的 Tree 对象上
    #mask_ids 也被修改，只留下被mask的非叶子节点
    flag = True #判断是否删除干净
    while flag:
        del_list =[]
        for mask_id  in mask_ids:
            node = tree.get_node(mask_id)
            if node != None:
                if node.is_leaf():  # 如果是叶子节点，直接删除
                    tree.remove_node(mask_id)
                    del_list.append(mask_id)
                else:
                    data_list = node.data
                    data_list[1]=0
        # 删除所有flag为0的叶子节点,直到叶子节点flag全为1
        i = 0
        for node in tree.leaves():
            if node.data[1] == 0:
                tree.remove_node(node.identifier)
                del_list.append(node.identifier)
                i+=1
        for del_id in del_list:
            mask_ids.remove(del_id)

        if i ==0:
            flag = False
            dic_father,dic_child=father_childDict(tree)

    return tree,dic_father,dic_child,mask_ids

#!/usr/bin/env python
# coding: utf-8
from random import randint

data_path = "..//data//train.csv"

def str2num(data,test=False):
    data[1] = float(data[1])
    data[2] = float(data[2])
    data[3] = float(data[3])
    data[5] = int(data[5])
    data[6] = int(data[6])
    data[13] = int(data[13]) 
    data[20] = int(data[20])
    data[21] = int(data[21])
    data[23] = int(data[23])
    data[25] = float(data[25])
    data[26] = int(data[26])
    data[27] = int(data[27])
    data[28] = int(data[28])
    data[29] = int(data[29])
    data[42] = int(data[42])
    if test==False:
        data[43] = int(data[43])

def label_encoder(results):
    # categorical column [4,7,8,9,10,11,12,14,15,16,17,18,19,22,24,30,31,32,33,34,35,36,37,38,39,40,41]
    cat = [4,7,8,9,10,11,12,14,15,16,17,18,19,22,24,30,31,32,33,34,35,36,37,38,39,40,41]
    Dict=dict()
        
    for col in cat:
        Dict[col] = dict()
        counter = 0          
        for r in results:
            if r[col] not in Dict[col].keys():
                Dict[col][r[col]]=counter
                counter += 1
            r[col] = Dict[col][r[col]]
    return Dict

def read_csv(path):
    with open(path , 'r') as f:
        header = []
        results = []
        for line in f:
            if header == []:
                header = line.split(',')
                header[-1] = header[-1][:-1]
            else:
                words = line.split(',')
                words[-1] = words[-1][:-1]
                str2num(words)
                results.append(words)
        label_encoder(results)
        return header,results

def train_test_split(data,val_size = 0.1, test_size = 0.1):
    train = int(len(data)*(1-val_size-test_size))
    val = int(len(data)*(1-test_size))
    
    data = shuffle(data)
    train_set = data[:train]
    val_set = data[train:val]
    test_set = data[val:]
    return train_set,val_set,test_set

def shuffle(data):
    length = len(data)
    for i in range(int(length*1.5)):
        ind1 = randint(0,length-1)
        ind2 = randint(0,length-1)
        temp = data[ind1]
        data[ind1] = data[ind2]
        data[ind2] = temp
    return data

def data2XY(data):
    X = []
    y = []
    for d in data:
        X.append(d[1:-1]) # ID is excluded
        y.append(d[-1])
    return X,y
#!/usr/bin/env python
# coding: utf-8

def str2num(data):
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
    data[43] = int(data[43])

def label_encoder(results):
    # categorical column [4,7,8,9,10,11,12,14,15,16,17,18,19,22,24,30,31,32,33,34,35,36,37,38,39,40,41]
    cat = [4,7,8,9,10,11,12,14,15,16,17,18,19,22,24,30,31,32,33,34,35,36,37,38,39,40,41]
    for col in cat:
        Dict = dict()
        counter = 0
        for r in results:
            if r[col] not in Dict.keys():
                Dict[r[col]]=counter
                counter += 1
            r[col] = Dict[r[col]]

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





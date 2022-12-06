#!/usr/bin/env python
# coding: utf-8

from seaborn import heatmap
from matplotlib.pyplot import show,xlabel,ylabel
from data import read_csv,train_test_split,data2XY

def visualize_score(conf_mat):
    conf_mat_flatten = [conf_mat[0][0],conf_mat[0][1],conf_mat[1][0],conf_mat[1][1]]
    names = ['TN','FP','FN','TP']
    counts = ["{0:0.0f}".format(value) for value in conf_mat_flatten]
    total = conf_mat[0][0]+conf_mat[0][1]+conf_mat[1][0]+conf_mat[1][1]
    percentages = ["{0:.2%}".format(value/total) for value in conf_mat_flatten]
    labels = [f"{n}\n{c}\n{p}" for n, c, p in zip(names,counts,percentages)]
    labels = [[labels[0],labels[1]],[labels[2],labels[3]]]
    heatmap(conf_mat, annot=labels, fmt='')
    xlabel('Predict', fontsize = 15) # x-axis label with fontsize 15
    ylabel('Truth', fontsize = 15) # y-axis label with fontsize 15

def data_process(path=  "..//data//train.csv"):
    header,data,Dict = read_csv(path)
    train,test,_ = train_test_split(data,val_size=0.2,test_size=0)
    X_train,y_train = data2XY(train)
    X_test,y_test = data2XY(test)
    return X_train,y_train,X_test,y_test,Dict

def confusion_matrix(y_true,y_pred):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i,j in zip(y_true,y_pred):
        if i==j:
            if i == 1:
                TP += 1
            else:
                TN += 1
        else:
            if j == 1:
                FP += 1
            else:
                FN += 1
    if TP==0:
        recall = 0
        precision = 0
    else:
        recall = TP/(TP+FN)
        precision = TP/(TP+FP)
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    print("Recall = {0:.2%}".format(recall))
    print("Precision = {0:.2%}".format(precision))
    print("Accuracy = {0:.2%}".format(accuracy))
    return [[TN,FP],[FN,TP]]

def classify(clf,name,data_path =  "..//data//train.csv"):
    print(f"{name}:")
    X_train,y_train,X_test,y_test,Dict = data_process(data_path)
    
    if "my" in name: #My random forest classifier
        clf.set_Dict(Dict)
    
    if "Cat" in name: # CatBoost
        clf.fit(X_train, y_train,verbose=0)
    else:
        clf.fit(X_train, y_train)
     
    y_pred = clf.predict(X_train)
    conf_mat = confusion_matrix(y_train,y_pred)
    visualize_score(conf_mat)
    show()
    
    y_pred = clf.predict(X_test)
    conf_mat = confusion_matrix(y_test,y_pred)
    visualize_score(conf_mat)
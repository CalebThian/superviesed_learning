#!/usr/bin/env python
# coding: utf-8

# In[3]:


import seaborn as sns
import matplotlib.pyplot as plt
def visualize_score(conf_mat):
    conf_mat_flatten = [conf_mat[0][0],conf_mat[0][1],conf_mat[1][0],conf_mat[1][1]]
    names = ['TN','FP','FN','TP']
    counts = ["{0:0.0f}".format(value) for value in conf_mat_flatten]
    total = conf_mat[0][0]+conf_mat[0][1]+conf_mat[1][0]+conf_mat[1][1]
    percentages = ["{0:.2%}".format(value/total) for value in conf_mat_flatten]
    labels = [f"{n}\n{c}\n{p}" for n, c, p in zip(names,counts,percentages)]
    labels = [[labels[0],labels[1]],[labels[2],labels[3]]]
    sns.heatmap(conf_mat, annot=labels, fmt='')
    plt.xlabel('Predict', fontsize = 15) # x-axis label with fontsize 15
    plt.ylabel('Truth', fontsize = 15) # y-axis label with fontsize 15


# In[ ]:





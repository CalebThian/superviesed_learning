#!/usr/bin/env python
# coding: utf-8

# In[27]:


from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
from utils import data_process,visualize_score,confusion_matrix
from matplotlib.pyplot import show


# In[30]:


def classify(clf,name,data_path =  "..//data//train.csv"):
    print(f"{name}:")
    X_train,y_train,X_test,y_test = data_process(data_path)
    clf.fit(X_train, y_train,verbose=0)
    y_pred = clf.predict(X_train)
    conf_mat = confusion_matrix(y_train,y_pred)
    visualize_score(conf_mat)
    show()
    
    y_pred = clf.predict(X_test)
    conf_mat = confusion_matrix(y_test,y_pred)
    visualize_score(conf_mat)


# In[31]:


clf = XGBClassifier()
classify(clf,"XGBoost Classifier")


# In[32]:


clf = CatBoostClassifier()
classify(clf,"CatBoost Classifier")


# In[33]:


clf = lgb.LGBMClassifier(is_unbalance=True)
classify(clf,"Light Boost Classifier")


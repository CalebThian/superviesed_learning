#!/usr/bin/env python
# coding: utf-8

# In[1]:


from naivebayes import NaiveBayesClassifier
from randomforest import myRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
from utils import classify,classify_k_fold


# In[2]:


k = [3,5,10]


# In[3]:


clf = NaiveBayesClassifier()
name = "Naive Bayes Clasifier"
classify(clf,name)
classify_k_fold(clf,name,k)


# In[4]:


clf = myRandomForestClassifier()
name = "my Random Forest Classifier"
classify(clf,name)
classify_k_fold(clf,name,k)

# In[6]:


clf = RandomForestClassifier(n_estimators = 10,max_depth=10, random_state=0)
name = "Random Forest Classifier of scikit-learn"
classify(clf,name)
classify_k_fold(clf,name,k)


# In[7]:


clf = XGBClassifier()
name = "XGBoost Classifier"
classify(clf,name)
classify_k_fold(clf,name,k)


# In[8]:


clf = CatBoostClassifier(verbose=0)
name = "Catboost Classifier"
classify(clf,name)
classify_k_fold(clf,name,k)


# In[9]:


clf = lgb.LGBMClassifier(is_unbalance=True)
name = "LightGBM Classifier"
classify(clf,name)
classify_k_fold(clf,name,k)


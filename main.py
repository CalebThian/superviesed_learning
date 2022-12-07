#!/usr/bin/env python
# coding: utf-8

from naivebayes import NaiveBayesClassifier
from randomforest import myRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
from utils import classify,classify_k_fold

k = [3,5,10]

clf = NaiveBayesClassifier()
name = "Naive Bayes Clasifier"
classify(clf,name)
classify_k_fold(clf,name,k)

clf = myRandomForestClassifier()
name = "my Random Forest Classifier"
classify(clf,name)
classify_k_fold(clf,name,k)

clf = RandomForestClassifier(n_estimators = 10,max_depth=10, random_state=0)
name = "Random Forest Classifier of scikit-learn"
classify(clf,name)
classify_k_fold(clf,name,k)

clf = XGBClassifier()
name = "XGBoost Classifier"
classify(clf,name)
classify_k_fold(clf,name,k)

clf = CatBoostClassifier(verbose=0)
name = "Catboost Classifier"
classify(clf,name)
classify_k_fold(clf,name,k)

clf = lgb.LGBMClassifier(is_unbalance=True)
name = "LightGBM Classifier"
classify(clf,name)
classify_k_fold(clf,name,k)


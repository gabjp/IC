#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 07:49:29 2022

@author: gjperin
"""

from useful import display_metrics, i_values
from data import train_with_WISE, train_no_WISE, data

import numpy as np
import pandas as pd
from time import perf_counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate

scoring = {
    'Precision_QSO': make_scorer(precision_score, average=None, labels=[0]),
    'Recall_QSO': make_scorer(recall_score, average=None, labels=[0]),
    'f1_QSO': make_scorer(f1_score, average=None, labels=[0]),
    'Precision_STAR': make_scorer(precision_score, average=None, labels=[1]),
    'Recall_STAR': make_scorer(recall_score, average=None, labels=[1]),
    'f1_STAR': make_scorer(f1_score, average=None, labels=[1]),
    'Precision_GAL': make_scorer(precision_score, average=None, labels=[2]),
    'Recall_GAL': make_scorer(recall_score, average=None, labels=[2]),
    'f1_GAL': make_scorer(f1_score, average=None, labels=[2]),
    'f1_macro': make_scorer(f1_score, average='macro')
}

no_wise = data[data["final_train"]==1].drop(columns=["final_train","wise","w1mpro", "w2mpro", 'cv_1', 'cv_2', 'cv_3', 'cv_4', 'cv_5'])
no_wise.index = [t for t in range(len(no_wise.index))]
y = no_wise.pop("target")
kf = StratifiedKFold(n_splits=5, random_state=2, shuffle=True) 


#clf = RandomForestClassifier(random_state=2, n_estimators=100, bootstrap=False)
#RGS:
clf = RandomForestClassifier(random_state=2,n_estimators= 200, min_samples_split= 2, min_samples_leaf= 1, max_features= 5, max_depth= 90, bootstrap= False)
results = cross_validate(estimator=clf, X=no_wise, y=y, cv=kf, scoring=scoring, return_train_score=False)

print("12s+4M+*:")
#print("12s+4M+:")
for key in results:
    print(f"{key}: {np.round(np.mean(results[key]*100),2)}+-{np.round(np.std(results[key]*100),2)}")




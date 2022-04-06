#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 10:05:58 2022

@author: gjperin
"""

from data import train_with_WISE, train_no_WISE, data
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score

x_train_all = data[data["final_train"]==1].drop(columns=["final_train","wise","w1mpro", "w2mpro", 'cv_1', 'cv_2', 'cv_3', 'cv_4', 'cv_5'])
x_train_with_WISE = train_with_WISE.drop(columns=['cv_1','cv_2', 'cv_3', 'cv_4', 'cv_5', 'final_train', 'wise'])

y_train_all = x_train_all.pop("target")
y_train_with_WISE = x_train_with_WISE.pop("target")

random_grid = {'n_estimators': [100,200,300,400],
               'min_samples_split': [2,5], # Retirei None
               'min_samples_leaf': [1,2],  # Retirei None
               'max_features': ["sqrt", 5, 10, 15], 
               'max_depth': [i*10 for i in range(1,12)]+[None],
               'bootstrap': [True, False]}
rf = RandomForestClassifier(random_state=2)
rrf = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter =100, cv=3, verbose=1,random_state=2, n_jobs=-1, scoring = make_scorer(f1_score, average='macro'))

rrf.fit(x_train_all, y=y_train_all)
print("12S+4M")
print(rrf.best_params_)

print("12S+4M+2W")
rrf.fit(x_train_with_WISE, y=y_train_with_WISE)
print(rrf.best_params_)
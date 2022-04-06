#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 20:35:16 2022

@author: gjperin
"""

from useful import show_metrics, i_values
from data import train_no_WISE, train_with_WISE, test_no_WISE, test_with_WISE, tentativa

import numpy as np
import pandas as pd
from time import perf_counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

x_train_no_WISE=train_no_WISE.drop(columns=['w1mpro', 'w2mpro', 'cv_1','cv_2', 'cv_3', 'cv_4', 'cv_5', 'final_train', 'wise'])
x_train_with_WISE=train_with_WISE.drop(columns=[ 'cv_1','cv_2', 'cv_3', 'cv_4', 'cv_5', 'final_train', 'wise'])
x_test_no_WISE=test_no_WISE.drop(columns=['w1mpro', 'w2mpro', 'cv_1','cv_2', 'cv_3', 'cv_4', 'cv_5', 'final_train', 'wise'])
x_test_with_WISE=test_with_WISE.drop(columns=[ 'cv_1','cv_2', 'cv_3', 'cv_4', 'cv_5', 'final_train', 'wise'])

print(tentativa.equals(x_test_with_WISE))

sets = [(x_train_with_WISE,x_test_with_WISE),
        (pd.concat([x_train_with_WISE,x_train_no_WISE]).drop(columns=["w1mpro","w2mpro"]),x_test_with_WISE.drop(columns=['w1mpro', 'w2mpro'])),
        (pd.concat([x_train_with_WISE,x_train_no_WISE]).drop(columns=["w1mpro","w2mpro"]),x_test_no_WISE),
        (pd.concat([x_train_with_WISE,x_train_no_WISE]).drop(columns=["w1mpro","w2mpro"]),
     pd.concat([x_test_with_WISE,x_test_no_WISE]).drop(columns=["w1mpro","w2mpro"]))]

for train, test in sets:
        
    y_train = train["target"]
    x_train = train.drop(columns="target")
    y_test = test["target"]
    x_test= test.drop(columns="target")
    
    
    rf = RandomForestClassifier(random_state=2, n_estimators=100, bootstrap=True)
    
    tic = perf_counter()
    rf.fit(x_train, y_train )
    tac = perf_counter()
    pred = rf.predict(x_test)
    toc = perf_counter()
        
    results = confusion_matrix(y_test , pred)
    fit_time = tac-tic
    pred_time = toc-tac
    
    
    print(f"    Fit time: {np.round(fit_time,2)}")
    print(f"    Prediction time: {np.round(pred_time,2)}")
    show_metrics(results)
    print()
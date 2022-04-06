#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 14:42:01 2022

@author: gjperin
"""

from useful import display_metrics, i_values
from data import train_with_WISE

import numpy as np
import pandas as pd
from time import perf_counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC



#Tabela 3:


tab3 = train_with_WISE.drop(columns=["FWHM_n", "A", "B", "KRON_RADIUS", "final_train","wise"])

algs = ["RF_", "SVM_"]
experiments = ["12S", "12S+2W", "5S", "5S+2W"]
drops = [["w1mpro", "w2mpro"],[],["w1mpro", "w2mpro","F378_iso", "F395_iso", "F410_iso", "F430_iso", "F515_iso", "F660_iso", "F861_iso"],
         ["F378_iso", "F395_iso", "F410_iso", "F430_iso", "F515_iso", "F660_iso", "F861_iso"]]

i_results = {}


for alg in range(1,2):

    for j in range(4):
    
        useful = tab3.drop(columns=drops[j])
            
        results = []
        fit_time = []
        pred_time = []
            
        for i in range(1,6):
            
            x_1 = useful.drop(columns=[ f"cv_{t}" for t in range(1,6) if t != i])
            fit_time
            x_1_train = x_1[x_1[f"cv_{i}"]==1].drop(columns=f"cv_{i}")
            x_1_test = x_1[x_1[f"cv_{i}"]==0].drop(columns=f"cv_{i}")
            
            y_1_train = x_1_train.pop("target")
            y_1_test = x_1_test.pop("target")
                
            if alg:
                mod = SVC(decision_function_shape="ovo", kernel="rbf", random_state=2)
            else:
                mod= RandomForestClassifier(bootstrap=False, random_state=2)
            tic = perf_counter()
            mod.fit(x_1_train, y_1_train)
            tac = perf_counter()
            pred = mod.predict(x_1_test)
            toc = perf_counter()
                
            results.append(confusion_matrix(y_1_test, pred))
            fit_time.append(tac-tic)
            pred_time.append(toc-tac)
            
        if __name__ == "__main__":
            print(f"Experimento: {algs[alg]+experiments[j]}")
            print(f"    Fit time: {np.round(np.mean(fit_time),2)}+-{np.round(np.std(fit_time),2)}")
            print(f"    Prediction time: {np.round(np.mean(pred_time),2)}+-{np.round(np.std(pred_time),2)}")
            display_metrics(results)
            print()
        else:
            i_results[algs[alg]+experiments[j]] = i_values(results)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 17:46:00 2022

@author: gjperin
"""

from useful import display_metrics, i_values
from data import train_with_WISE

import numpy as np
import pandas as pd
from time import perf_counter

from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler



#Tabela 3:


tab3 = train_with_WISE.drop(columns=["FWHM_n", "A", "B", "KRON_RADIUS", "final_train","wise"])

experiments = ["12S", "12S+2W", "5S", "5S+2W"]
drops = [["w1mpro", "w2mpro"],[],["w1mpro", "w2mpro","F378_iso", "F395_iso", "F410_iso", "F430_iso", "F515_iso", "F660_iso", "F861_iso"],
         ["F378_iso", "F395_iso", "F410_iso", "F430_iso", "F515_iso", "F660_iso", "F861_iso"]]

ip_results = {}


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
            
        ss = StandardScaler()
        ss.fit(x_1_train)
        x_1_train = ss.transform(x_1_train)
        x_1_test = ss.transform(x_1_test)
       
        mod = SVC(decision_function_shape="ovo", kernel="rbf", random_state=2)
        
        tic = perf_counter()
        mod.fit(x_1_train, y_1_train)
        tac = perf_counter()
        pred = mod.predict(x_1_test)
        toc = perf_counter()
            
        results.append(confusion_matrix(y_1_test, pred))
        fit_time.append(tac-tic)
        pred_time.append(toc-tac)
        
    if __name__ == "__main__":
        print(f"Experimento: SVM_{experiments[j]} (padronizado)")
        print(f"    Fit time: {np.round(np.mean(fit_time),2)}+-{np.round(np.std(fit_time),2)}")
        print(f"    Prediction time: {np.round(np.mean(pred_time),2)}+-{np.round(np.std(pred_time),2)}")
        display_metrics(results)
        print()
    else:
        ip_results[f"SVM_{experiments[j]}_P"] = i_values(results)
        
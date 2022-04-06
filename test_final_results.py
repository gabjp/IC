#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 08:22:33 2022

@author: gjperin
"""


from useful import show_metrics, i_values
from data import train_no_WISE, train_with_WISE, test_no_WISE, test_with_WISE, tentativa

import numpy as np
import pandas as pd
from time import perf_counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

import pickle

print(tentativa.equals(test_with_WISE[tentativa.columns]))
    
y_test = tentativa["target"]
x_test= tentativa.drop(columns="target")
print(x_test.columns)


with open("data/iDR3n4_RF_12S2W4M", "rb") as arq:
    rf = pickle.load(arq)
pred = rf.predict(x_test)

print(pred)
print(y_test)

    
results = confusion_matrix(y_test , pred)
print(results)
show_metrics(results)
print()
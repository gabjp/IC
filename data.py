#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 15:09:28 2022

@author: gjperin
"""


import numpy as np
import pandas as pd


data = pd.read_csv("data/Spectroscopic_sample.csv")
data = data.drop(columns=["RA", "DEC"])

with_WISE = data[data["wise"]==1]
test_with_WISE = with_WISE[with_WISE["final_train"] == 0]
train_with_WISE = with_WISE[with_WISE["final_train"] == 1]

no_WISE = data[data["wise"]==0]
test_no_WISE = no_WISE[no_WISE["final_train"] == 0]
train_no_WISE = no_WISE[no_WISE["final_train"] == 1]

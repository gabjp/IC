#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 11:43:10 2022

@author: gjperin
"""

import numpy as np
from statistics import harmonic_mean


def p_r(conf):
    # Dada uma matriz de confusão, calcular precision(p) e recall(r) para cada obejeto
    
    results = []
    
    for i in range(3):
    
        p_i = conf[i,i]/np.sum(conf[:,i])
        r_i = conf[i,i]/np.sum(conf[i,:])
        results.append(p_i)
        results.append(r_i)
        
    return results

def fi(pr):
    #Dado valores de p e r para cada objeto calcular o valor f
    
    fs = []
    for i in range(0,6,2):
        fs.append(harmonic_mean([pr[i], pr[i+1]]))
    return fs
        
def metrics(mat):
    #Dada uma matriz de confusão, devolver todos os valores de p_i, r_i, f_i e f
    
    pr = p_r(mat)
    fs = fi(pr)
    f = np.mean(fs)
    return pr+fs+[f] # [p_0, r_0,p_1, r_1,p_2, r_2, f_0, f_1, f_2, f]

def ametrics(listmat):
    
    values = []
    
    for mat in listmat:
        values.append(metrics(mat))
        
    values = np.array(values)
    mean = np.round(values.mean(axis=0)*100,2) 
    std = np.round(values.std(axis=0)*100,2) 
    return mean, std

def display_metrics(listmat):
    mean, std = ametrics(listmat)
    print(f"""    F-mean: {mean[9]}+-{std[9]}
    P_QSO: {mean[0]}+-{std[0]}
    P_STAR: {mean[2]}+-{std[2]}
    P_GAL: {mean[4]}+-{std[4]}
    R_QSO: {mean[1]}+-{std[1]}
    R_STAR: {mean[3]}+-{std[3]}
    R_GAL: {mean[5]}+-{std[5]}
    F_QSO: {mean[6]}+-{std[6]}
    F_STAR: {mean[7]}+-{std[7]}
    F_GAL: {mean[8]}+-{std[8]}
    """)
    
def i_values(listmat):
    #Dada uma lista de matrizes de confusão de um experimento, devolver um dict de listas com cada um dos valores individuais.
    resultados = {"F-mean": [],
    "P_QSO": [],
    "P_STAR": [],
    "P_GAL": [],
    "R_QSO": [],
    "R_STAR": [],
    "R_GAL": [],
    "F_QSO": [],
    "F_STAR": [],
    "F_GAL": []}
    
    keys = ["P_QSO", "R_QSO", "P_STAR", "R_STAR", "P_GAL", "R_GAL", "F_QSO", "F_STAR", "F_GAL", "F-mean"]
    
    for mat in listmat:
        x = metrics(mat)
        for i in range(10):
            resultados[keys[i]].append(x[i])
            
    return resultados

def show_metrics(mat):
    #Dada uma matriz de confusão, printar as metricas.
    
    m = metrics(mat)
    print(f"""    F-mean: {np.round(m[9]*100,2)}
    P_QSO: {np.round(m[0]*100,2)}
    P_STAR: {np.round(m[2]*100,2)}
    P_GAL: {np.round(m[4]*100,2)}
    R_QSO: {np.round(m[1]*100,2)}
    R_STAR: {np.round(m[3]*100,2)}
    R_GAL: {np.round(m[5]*100,2)}
    F_QSO: {np.round(m[6]*100,2)}
    F_STAR: {np.round(m[7]*100,2)}
    F_GAL: {np.round(m[8]*100,2)}
    """)
        
        
        
        
        
    
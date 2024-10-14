import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np

def compute_AIC_component(X, cov):
    cov_inv = np.linalg.pinv(cov)
    return cov_inv - cov_inv @ X @ np.linalg.pinv(X.T @ cov_inv @ X) @ X.T @ cov_inv

def compute_AIC(X, y, cov):
    A_M = compute_AIC_component(X, cov)
    aic = y.T @ A_M @ y + 2 * X.shape[1]
    return aic

def compute_teststatistics(X, y, j_selected, cov):

    n = y.shape[0]

    ej = np.zeros(X.shape[1])
    ej[j_selected] = 1
        
    etaj = (np.linalg.pinv(X.T @ X) @ X.T).T @ ej
    z_obs = etaj.T @ y
    
    var = etaj.T @ cov @ etaj

    b = (cov @ etaj) / var
    a = (np.identity(n) - b.reshape(-1,1) @ etaj.reshape(1,-1)) @ y
    
    return a, b, z_obs, var
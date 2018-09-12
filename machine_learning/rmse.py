#coding:utf-8
'''
均方误差
'''
import numpy as np

y_hat = np.array([0 , 1 , 2])
y_true = np.array([0 , 0.5 , 1.5])

def rmse(preditions , targets):
    differences = preditions - targets
    differences_squared = differences ** 2
    mean_of_differences_squared = differences_squared.mean()
    rmse_val = np.sqrt(mean_of_differences_squared)
    return rmse_val

rmse_val = rmse(y_hat , y_true)
print('rms error is:' , rmse_val)
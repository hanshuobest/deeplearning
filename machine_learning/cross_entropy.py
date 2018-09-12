#coding:utf-8
'''
交叉熵损失
'''

import numpy as np
predictions = np.array([[0.25 , 0.25 , 0.25 , 0.25] ,
                        [0.01 , 0.01 , 0.01 , 0.01]])

print(np.log(predictions))
targets = np.array([[0 , 0 , 0 , 1] ,
                    [0 , 0 , 0 , 1]])

def cross_entropy(predictions , targets , epsilon=1e-10):
    predictions = np.clip(predictions , epsilon , 1. - epsilon)
    N = predictions.shape[0]
    print(targets * np.log(predictions + 1e-5))
    ce_loss = -np.sum(np.sum(targets * np.log(predictions + 1e-5)))/N
    return ce_loss

cross_entropy_loss = cross_entropy(predictions , targets)
print('cross_entropy_loss:' , cross_entropy_loss)

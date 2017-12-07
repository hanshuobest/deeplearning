import numpy as np

# 使用梯度上升找到最佳参数
def loadDataSet():
    dataMat = [] ; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0 , float(lineArr[0]) , float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat , labelMat

def sigmoid(X):
    return 1.0/(1 + np.exp(-X))

# 梯度上升更新权重
def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)             #convert to NumPy matrix
    labelMat = np.mat(classLabels).transpose() #convert to NumPy matrix
    m,n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n,1))
    for k in range(maxCycles):              #heavy on matrix operations
        h = sigmoid(dataMatrix*weights)     #matrix mult
        error = (labelMat - h)              #vector subtraction
        # yy = h.transpose() * (1 - h)
        # yy = yy[0 , 0]
        weights = weights + alpha *  dataMatrix.transpose()* error #matrix mult
    return weights

# 随机梯度上升
def stocGradAscent0(dataMatrix, classLabels):
    m,n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)   #initialize to all ones
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

dataArr , labelMat = loadDataSet()
# weights = gradAscent(dataArr , labelMat)
weights = stocGradAscent0(dataArr , labelMat)
print('weights:' , weights)

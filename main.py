import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import xlrd
import datetime

def loadDataSet(filename):
    workbook = xlrd.open_workbook(filename)
    worksheet = workbook.sheet_by_index(0)
    nrows = worksheet.nrows
    ncols = worksheet.ncols
    dataArr = []
    for i in range(nrows):
        rowVals = np.array(worksheet.row_values(i))
        dataArr.append(rowVals[1:])
    dataArr = np.array(dataArr)
    return dataArr

def Kmeans(dataArr, K, iters=10):
    sampNum, featNum = dataArr.shape
    # 假定K=3（3个聚类），首先从样本中随机选出k个作为初始均值向量
    mu = dataArr[np.random.randint(0, sampNum, K)]
    for i in range(iters):      # 设置repeat的次数
        C = [ list() for i in range(K)]
        for j in range(sampNum):
            diff = dataArr[j] - mu                      # 计算每个样本点到k个参考点的距离，执行次数K
            distance = [np.dot(diff[k], diff[k].T) for k in range(K)]   # 执行次数=样本数K
            # 选取距离的最小值进行分类
            classNum = np.argmin(distance)              # 执行次数K-1
            C[classNum].append(dataArr[j])
        C_Arr = np.array(C)
        muNew = np.array([np.mean(C_Arr[k], axis=0) for k in range(K)])
        if (muNew != mu).any():
            mu = muNew
        else:
            print("Kmeans algorithm converged in %d-th iteration" % (i+1))
            break

    plt.figure()
    for k in range(K):
        plt.scatter(np.array(C[k])[:,0], np.array(C[k])[:,1], c=list(mcolors.TABLEAU_COLORS)[k])
        plt.scatter(mu[k,0], mu[k,1], marker='*', c=list(mcolors.TABLEAU_COLORS)[k], s=100)
    curTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    figName = 'Kmeans_' + curTime + '.png'
    plt.savefig(figName)
    plt.show()


dataArr = loadDataSet('watermelon 4.0.xlsx')
Kmeans(dataArr, 3)
'''
dataArr = loadDataSet('testSet.xlsx')
Kmeans(dataArr, 4)
'''


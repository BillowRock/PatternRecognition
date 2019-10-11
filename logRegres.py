from numpy import *
def loadDataSet():    # 读样本集
    dataMat = []
    labelMat = []
    fr = open('testSet (line).txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return  dataMat, labelMat
def sigmoid(inX):             # sigmoid函数
    return 1.0/(1+exp(-inX))   # exp() 方法返回x的指数
def gradAscent(dataMatIn,classLabels):
    dataMatrix = mat(dataMatIn)      #转为矩阵
    labelMat = mat(classLabels).transpose()   #对标签进行转置
    m,n = shape(dataMatrix)      # 记录行列
    alpha = 1          # 步长
    maxCycles = 100           # 迭代次数
    weights = ones((n, 1))        # n*1的W向量 初值1
    for k in range(maxCycles):   # 迭代
        h = sigmoid(dataMatrix*weights)    # 利用sigmoid函数对判别结果输出，进行分类，h为n*1列向量，每个元素偏向1或者0
        error = (labelMat-h)                # 实际标签-分类后标签  
        weights = weights + alpha * dataMatrix.transpose()*error   # 按差值方向对W进行修正
        if k == 0 :listWeight = weights
        else:
            listWeight = hstack((listWeight,weights))

    plotWeights(listWeight,maxCycles)
    return weights
def plotWeights(listwei,cycles):
    import matplotlib.pyplot as plt
    list0 = listwei[0].transpose()
    list1 = listwei[1].transpose()
    list2 = listwei[2].transpose()
    x = range(0, cycles)
    #折线图绘制函数
    plt.plot(x,list0)
    plt.plot(x,list1)
    plt.plot(x,list2)
    #plt.show()
def plotBestFit(wei):   # 画图
    import matplotlib.pyplot as plt
    weights = wei
    dataMat,labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x= arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title("step 0.1,cycle 500")
    plt.show()
dataMat,labelMat = loadDataSet()
print(gradAscent(dataMat,labelMat))

plotBestFit(gradAscent(dataMat,labelMat).getA())

---
title: Adaboost
date: 2016-12-20 00:35:57
---

* Adaboost
Adaboost是一种迭代算法，其核心思想是针对同一个训练集训练不同的分类器(弱分类器)，然后把这些弱分类器集合起来，构成一个更强的最终分类器（强分类器）。
<!--more-->
Adaboost过程：
1. 先通过对N个训练样本的学习得到第一个弱分类器；
2. 将分错的样本和其他的新数据一起构成一个新的N个的训练样本，通过对这个样本的学习得到第二个弱分类器 ；
3. 将1和2都分错了的样本加上其他的新样本构成另一个新的N个的训练样本，通过对这个样本的学习得到第三个弱分类器；
4. 最终经过提升的强分类器。即某个数据被分为哪一类要由各分类器权值决定。

训练数据中的每个样本，并赋予其一个权重，这些权重构成了向量D。一开始，这些权重都初始化成相等值。首先在训练数据上训练出一个弱分类器并计算该分类器的错误率，然后在同一数据集上再次训练弱分类器。在分类器的第二次训练当中，将会重新调整每个样本的权重，其中第一次分对的样本的权重将会降低，而第一次分错的样本的权重将会提高。为了从所有弱分类器中得到最终的分类结果，AdaBoost为每个分类器都分配了一个权重值alpha，这些alpha值是基于每个弱分类器的错误率进行计算的。
错误率ε的定义为
$$ε=\frac{未正确分类的样本数}{所有样本数}$$
alpha：
$$α=\frac{1}{2}ln\frac{1-ε}{ε}$$
![](http://i1.piimg.com/567571/f46e6188192028b6.png)
AdaBoost算法的示意图。左边是数据集，其中直方图的不同宽度表示每个样例上的不同权重。在经过一个分类器之后，加权的预测结果会通过三角形中的alpha值进行加权。每个三角形中输出的加权结果在圆形中求和，从而得到最终的输出结果计算出alpha值之后，可以对权重向量D进行更新，以使得那些正确分类的样本的权重降低而错分样本的权重升高。
D的计算方法如下:
如果某个样本被正确分类，那么该样本的权重更改为：
$$D_i^{(t+1)}=\frac{D_i^{(t)}e^-α}{Sum(D)}$$
而如果某个样本被错分，那么该样本的权重更改为：
$$D_i^{(t+1)}=\frac{D_i^{(t)}e^α}{Sum(D)}$$
在计算出D之后，AdaBoost又开始进入下一轮迭代。Ad-aBoost算法会不断地重复训练和调整权重的过程，直到训练错误率为0或者弱分类器的数目达到用户的指定值为止。

* 自适应数据加载函数
```
def loadDataSet(filename):  
    numFeat = len(open(filename).readline().split('\t'))  
    dataMat = []  
    labelMat=[]  
    fr = open(filename)  
    for line in fr.readlines():  
        lineArr= []  
        curLine = line.strip('\n').split('\t')  
        for i in range(numFeat - 1):  
            lineArr.append(float(curLine[i]))  
        dataMat.append(lineArr)  
        labelMat.append(float(curLine[-1]))  
    fr.close()  
    return dataMat, labelMat
```
>并不指定每个文件中的特特征树木，可以自动检测，并假定最后一个特征是类别标签

* 单层决策树生成函数

```
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):  
    retArray = ones((shape(dataMatrix)[0],1))  
    if threshIneq == 'lt':  
        retArray[dataMatrix[:,dimen]<threshVal] = -1.0  
    else:  
        retArray[dataMatrix[:,dimen]<threshVal] = -1.0  
    return retArray  

def buildStump(dataArr, classLabels, D):  
    dataMatrix = mat(dataArr); labelMat = mat(classLabels).T  
    m,n = shape(dataMatrix)  
    numSteps = 10.0; bestStump = {}; #定义一个空字典，用于存储给定权重向量D时所得到的最佳单层决策树的相关信息  
	bestClassEst = mat(zeros((m,1)))  
    minError = inf  #最小错误率初始化为无穷大
    for i in range(n):  #在所有数据集的所有特征上遍历  
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max();  
        stepSize = (rangeMax - rangeMin)/numSteps #通过计算特征的最小值和最大值来计算步长，numSteps越大，步长越小   
        for j in range(-1, int(numSteps)+1):  #按分的步长总数进行循环
            for inequal in ['lt','gt']:  
                threshVal = (rangeMin + float(j)*stepSize)  
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)  
                errArr = mat(ones((m,1)))   #构建错误数组  errArr，如果predict-edVals中的值不等于labelMat中的真正类别标签值，那么errArr的相应位置为1
                errArr[predictedVals == labelMat] = 0  
                weightedError = D.T * errArr           #这里的error是错误向量errArr和权重向量D的相应元素相乘得到的即加权错误率  
                #print "split: dim %d, thresh %.2f, thresh inequal: %s, the weighted error is %.3f" %(i, threshVal, inequal, weightedError)  
                if weightedError < minError:  
                    minError = weightedError  
                    bestClassEst = predictedVals.copy()  
                    bestStump['dim'] = i  
                    bestStump['thresh'] = threshVal  
                    bestStump['ineq'] = inequal  
    return bestStump, minError, bestClassEst #返回分类的最小错误率
```
>stumpClassify()通过阈值比较进行分类，可以通过数组过滤实现，首先将返回的元素全部设置为1，不满足等式的元素设置为为-1。
>buildStump()有三层循环构建了单层决策树，最外层循环为遍历特征，次外层循环为遍历的步长，最内层为是否大于或小于阀值。构建的最小错误率为加权错误率，这就是为什么增加分错样本的权重，因为分错样本的权重增加了，下次如果继续分错，加权错误率会很大，这就不满足算法最小化加权错误率了。此外，加权错误率在每次迭代过程中一定是逐次降低的。
>单层决策树的生成函数是决策树的一个简化版本。它就是所谓的弱学习器，即弱分类算法。

* 基于单层决策树的adaboost训练过程  
```
def adaBoostTrainDS(dataArr, classLabels, numIt = 40):  
    weakClassArr = []  #建立一个单层决策树数组
    m = shape(dataArr)[0]  #得到数据点的数目
    D = mat(ones((m,1))/m) #向量D非常重要，它包含了每个数据点的权重 
    aggClassEst = mat(zeros((m,1)))#列向量aggClassEst，记录每个数据点的类别估计累计值  
    for i in range(numIt):  
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)  #bestStump=字典,error=分类错误率,classEst=列向量，预测之后的分类列表
       # print "D:", D.T  
        alpha = float(0.5 * log((1.0 - error)/max(error, 1e-16)))   #确保在没有错误时不会发生除零溢出  
        bestStump['alpha'] = alpha  
        weakClassArr.append(bestStump)  
        #print "classEst:", classEst.T  
        expon = multiply(-1 * alpha * mat(classLabels).T, classEst)    #乘法用于区分是否正确或者错误样本,样本被正确分类的话expon为负，错误分类的话为正,其中第一次分对的样本的权重将会降低，而第一次分错的样本的权重将会提高  
        D = multiply(D, exp(expon))  #计算新权重向量D  
        D = D/D.sum()            # 归一化用的  
        aggClassEst += alpha * classEst    #累加变成强分类器  
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m,1)))  
        errorRate = aggErrors.sum()/m  
        print "total error: ", errorRate, "\n"  
        if errorRate == 0.0: break  
    return weakClassArr, aggClassEst  
```
>对每次迭代：
  利用buildStump()函数找到最佳的单层决策树
  将最佳单层决策树加入到单层决策树数组
  计算alpha
  计算新的权重向量D
  更新累计类别估计值
  如果错误率等于0.0，则退出循环
我们假定迭代次数设为9，如果算法在第三次迭代之后错误率为0，那么就会退出迭代过程，因此，此时就不需要执行所有的9次迭代过程。每次迭代的中间结果都会通过print语句进行输出。

* adaboost分类函数
```
def adaClassify(datToClass, classifierArr):  
    dataMatrix = mat(datToClass)  
    m = shape(dataMatrix)[0]  
    aggClassEst = mat(zeros((m,1)))  
    for i in range(len(classifierArr)):  
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])  
        aggClassEst += classifierArr[i]['alpha']*classEst  
        print aggClassEst  
    return sign(aggClassEst)  
```
>adaClassify()函数就是利用训练出的多个弱分类器进行分类的函数。该函数的输入是由一个或者多个待分类样例datToClass以及多个弱分类器组成的数组classifierArr。程序返回aggClassEst的符号，即如果aggClassEst大于0则返回+1，而如果小于0则返回-1。

* 画决策树的图 
```
def plot_Fig(xMat,yMat,weakClassArr):  
    xMat = mat(xMat)  
    fig = plt.figure()  
    ax = fig.add_subplot(111)  
    for i in range(len(yMat)):  
        if yMat[i] == -1.0: #如果标签为-1，则将数据点标为蓝色方块  
            ax.scatter(xMat[i,0],xMat[i,1],color='b',marker='s') #注意flatten的用法  
        else:  #如果标签为1，则将数据点标为红色圆形  
            ax.scatter(xMat[i,0],xMat[i,1],color='r',marker='o')  
    for i in range(len(weakClassArr)): #根据弱分类器数组画出决策树图形  
        if weakClassArr[i].get("dim") == 0:   
            y = arange(0.0,3.0,0.1)  
            x = weakClassArr[i].get("thresh") #得到阈值  
            x = repeat(x,len(y))  
            ax.plot(x,y)  
        if weakClassArr[i].get("dim") == 1:  
            x = arange(0.0,3.0,0.1)  
            y = weakClassArr[i].get("thresh")  
            y = repeat(y,len(x))  
            ax.plot(x,y)   
    plt.show()  
```
* ROC曲线
```
def plotROC(predStrengths, classLabels):  
    import matplotlib.pyplot as plt  
    cur = (1.0,1.0) #cursor绘制光标的位置  
    ySum = 0.0 #variable to calculate AUC；用于计算AUC的值  
    numPosClas = sum(array(classLabels)==1.0) #计算正例的数目  
    yStep = 1/float(numPosClas); #确定y坐标轴上的步长，因为当y为1时，对应的正例个数为numPosClas  
    xStep = 1/float(len(classLabels)-numPosClas) #计算x坐标轴上的步长，因为当x为1时，对应的负例个数为总数减去numPosClas  
    sortedIndicies = predStrengths.argsort()#get sorted index, it's reverse  
    fig = plt.figure()  
    fig.clf()  
    ax = plt.subplot(111)  
    #loop through all the values, drawing a line segment at each point  
    for index in sortedIndicies.tolist()[0]: #利用tolist()转化为列表，  
        if classLabels[index] == 1.0: #每得到一个标签为1.0的类，沿着y轴的方向下降一个步长，即不断降低真阳率（好好体会为什么这样做）  
            delX = 0; delY = yStep;  
        else:  
            delX = xStep; delY = 0; #类似   
            ySum += cur[1] #先对所有矩形的高度进行累加（当y值下降时不累加），最后再乘以xStep就是其总面积。  
        #draw line from cur to (cur[0]-delX,cur[1]-delY)  
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY], c='b')  
        cur = (cur[0]-delX,cur[1]-delY) #更新绘制光标的位置  
    ax.plot([0,1],[0,1],'b--')  
    plt.xlabel('False positive rate'); plt.ylabel('True positive rate')  
    plt.title('ROC curve for AdaBoost horse colic detection system')  
    ax.axis([0,1,0,1])  
    print "the Area Under the Curve is: ",ySum*xStep  
    plt.show()
```
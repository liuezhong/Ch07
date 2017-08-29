import adaboost
from numpy import *

# dataMat, classLabels = adaboost.loadSimpData()
# D = mat(ones((5,1))/5)
# print(adaboost.buildStump(dataMat, classLabels, D))

# classifierArray = adaboost.adaBoostTrainDS(dataMat, classLabels, 9)
# print(classifierArray)

# print(adaboost.adaClassify([0,0],classifierArray))

# print(adaboost.adaClassify([[5,5],[0,0]],classifierArray))

# dataArr, labelArr = adaboost.loadDataSet('horseColicTraining2.txt')
# classifierArray = adaboost.adaBoostTrainDS(dataArr, labelArr, 10)
# testArr, testLabelArr = adaboost.loadDataSet('horseColicTest2.txt')
# prediction10 = adaboost.adaClassify(testArr, classifierArray)
# errArr = mat(ones((67,1)))
# print(errArr[prediction10!=mat(testLabelArr).T].sum())

dataArr, labelArr = adaboost.loadDataSet('horseColicTraining2.txt')
classifierArray,aggClassEst = adaboost.adaBoostTrainDS(dataArr, labelArr,10)
adaboost.plotROC(aggClassEst.T,labelArr)
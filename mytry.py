import adaboost
from numpy import *

dataMat, classLabels = adaboost.loadSimpData()
D = mat(ones((5,1))/5)
# print(adaboost.buildStump(dataMat, classLabels, D))

classifierArray = adaboost.adaBoostTrainDS(dataMat, classLabels, 9)
print(classifierArray)
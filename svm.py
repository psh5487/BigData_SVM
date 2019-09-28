import numpy as np
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkContext

sc = SparkContext()

#load and parse the data
def parsePoint(line):
        values = [np.float(x) for x in line.replace(',', ' ').split(' ')]
	return LabeledPoint(values[0], values[1:])

traindata = sc.textFile("/user/cloudera/hw2/train_hw2.csv")
testdata = sc.textFile("/user/cloudera/hw2/test_hw2.csv")

parsedTrainData = traindata.map(parsePoint)
parsedTestData = testdata.map(parsePoint)

#Build the model with train data
model = SVMWithSGD.train(parsedTrainData, iterations=100, step=1.0, regParam=0.01, regType="l2")

#Evaluate the model on test data
labelsAndPreds = parsedTestData.map(lambda p: (p.label, model.predict(p.features)))
a = labelsAndPreds.filter(lambda lp: lp[0]==1 and lp[1]==1).count()
b = labelsAndPreds.filter(lambda lp: lp[0]==1 and lp[1]==0).count()
c = labelsAndPreds.filter(lambda lp: lp[0]==0 and lp[1]==1).count()
d = labelsAndPreds.filter(lambda lp: lp[0]==0 and lp[1]==0).count()

print("Confusion Matrix: ")
print("TP = " + str(a))
print("FN = " + str(b))
print("FP = " + str(c))
print("TN = " + str(d))
print("\n")

#Calculation
a = np.float(a)
b = np.float(b)
c = np.float(c)
d = np.float(d)

accuracy = (a+d) / (a+b+c+d)
precision = a / (a+c)
recall = a / (a+b)
f1 = 2*a / (2*a+b+c)

print('Accuracy: %f' %accuracy)
print('Precision: %f' %precision)
print('Recall: %f' %recall)
print('F1: %f' %f1)

#save and load model
model.save(sc, "/user/cloudera/hw2/results/2015310884_SVM")
sameModel = SVMModel.load(sc, "/user/cloudera/hw2/results/2015310884_SVM")




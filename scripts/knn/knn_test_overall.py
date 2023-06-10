import pandas as pd
import numpy as np
import time as time
from sklearn.neighbors import KNeighborsClassifier

trainingSamples = 50000
testingSamples = 10000

startTrainingTime = 0
endTrainingTime = 0
trainingTime = 0

startTestingTime = 0
endTestingTime = 0
testingTime = 0

def loadDataset(fileName, samples):
    x = []
    y = []
    train_data = pd.read_csv(fileName)
    y = np.array(train_data.iloc[0:samples, 0])
    x = np.array(train_data.iloc[0:samples, 1:]) / 255
    return x, y

def main():
    train_x, train_y = loadDataset("../../datasets/mnist/mnist_train.csv", trainingSamples)
    test_x, test_y = loadDataset("../../datasets/mnist/mnist_test.csv", testingSamples)
    clf = KNeighborsClassifier()
    startTrainingTime = time.time()
    clf.fit(train_x, train_y)
    endTrainingTime = time.time()
    trainingTime = endTrainingTime - startTrainingTime

    startTestingTime = time.time()
    predictions = clf.predict(test_x)
    endTestingTime = time.time()
    testingTime = endTestingTime - startTestingTime

    overallAccuracy = (predictions == test_y).mean() * 100

    print("-------------------------------")
    print("Results")
    print("-------------------------------")
    print("Training samples: ", trainingSamples)
    print("Training time: ", round(trainingTime, 2), " s")
    print("Testing samples: ", testingSamples)
    print("Testing time: ", round(testingTime, 2), " s")
    print("Overall accuracy: ", round(overallAccuracy, 2), "%")

if __name__ == "__main__":
    main()

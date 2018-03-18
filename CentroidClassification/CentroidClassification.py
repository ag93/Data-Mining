from math import *

import numpy as np

import DataHandler as dh

import random

import matplotlib.pyplot as plt

def centroid_classifier(train, test):
    trainX = train[1:]
    trainY = train[0]

    testX = test[1:]
    testY = test[0]

    testX = np.array(testX).transpose()

    centroid = centroid_calculate(trainX,trainY)
    count = 0
    for i in range(len(testX)):

        predict = classify(centroid[1], centroid[0], testX[i])
        if predict == testY[i]:
            count += 1

    return count/len(testX)


def centroid_calculate(X, Y):
    centroid = {}
    label = []
    means = []
    # print("label:",Y)
    data = np.array(X).transpose()
    for i in range(len(Y)):
        if Y[i] in centroid:
            centroid[Y[i]].append(data[i])
        else:
            centroid[Y[i]] = [data[i]]

    for key in centroid.keys():
        key_data = centroid[key]
        label.append(key)
        means.append(calculate_average(key_data))

    return [label, means]



def calculate_average(data):
    data = np.array(data).transpose().astype(np.float)

    n = len(data[0])
    mean = []
    for row in data:
        mean.append(sum(row)/n)

    return mean


def classify(X, Y, x):
    distance = []

    for mean in X:
        distance.append(euclidean_distance(mean, x))

    nearest_centroid_index = distance.index(min(distance))

    return Y[nearest_centroid_index]


def euclidean_distance(x, y):
    sum = 0
    for i in range(len(x)):
        sum += (float(x[i]) - float(y[i]))**2

    return sqrt(sum)


def format_data(filename, class_ids, test_instances):
    data = dh.pickDataClass(filename, class_ids)

    number_per_class = data[0].count(class_ids[0])
    trainX, trainY, testX, testY = dh.splitData2TestTrain(data, number_per_class, test_instances)

    dh.write_2_file(trainX, trainY, testX, testY)


# This function is used to perform K-fold cross validation
# k_fold: number of splits for cross-validation, train: data for CV
def cross_validation(k_fold, data):
    data = np.array(data).transpose().tolist()
    random.shuffle(data)
    n = len(data)
    len_k = n // k_fold
    accuracy_list = []
    for i in range(k_fold):
        start = i * len_k
        end = (i + 1) * len_k
        test = data[start:end]
        train = [x for x in data if x not in test]

        train = np.array(train).transpose().tolist()
        test = np.array(test).transpose().tolist()
        accuracy = centroid_classifier(train, test)
        accuracy_list.append(accuracy)
        print("Iterantion", i+1, "accuracy:", accuracy)

    print("Average accuracy:", sum(accuracy_list)/len(accuracy_list))


def start(filename, class_ids, test_instances):

    format_data(filename,class_ids, test_instances)

    train = []
    test = []
    with open('TrainingData.txt') as f:
        for line in f:
            data = line[:-1].split(',')
            train.append(data)

    with open('TestingData.txt') as f:
        for line in f:
            data = line[:-1].split(',')
            test.append(data)

    cross_validation(2, train)

    accuracy = centroid_classifier(train, test)
    #print("Overall Accuracy:", accuracy)
    return accuracy


if __name__ == "__main__":

    # filename = input("Enter filename: ")
    filename = "HandWrittenLetterss.txt"

    # class_ids = [1,2,3,4,5,6,7,8,9,10]
    # class_ids = random.sample(range(1, 27), 5)
    class_ids = [x for x in range(1,11)]
    print("For class:", class_ids)
    test_instances = [7,9]

    start(filename, class_ids, test_instances)


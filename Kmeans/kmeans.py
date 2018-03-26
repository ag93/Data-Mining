# -*- coding: utf-8 -*-
import DataHandler as dh
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.utils.linear_assignment_ import linear_assignment
import numpy as np

def start(filename, class_ids, test_instances):
    data = dh.pickDataClass(filename, class_ids)
    number_per_class = data[0].count(class_ids[0])
    trainX, trainY, testX, testY = dh.splitData2TestTrain(data, number_per_class, test_instances)
    dh.write_2_file(trainX, trainY, testX, testY)

    train = []
    with open('TrainingData.txt') as f:
        for line in f:
            data = line[:-1].split(',')
            train.append(data)
    
    X = np.array(train[1:]).transpose().tolist()
    kmeans = KMeans(n_clusters = 26)    
    kmeans.fit(X)
    Kmeans_labels = kmeans.labels_

    C = confusion_matrix(trainY, Kmeans_labels)
    C = C.T
    ind = linear_assignment(-C)
    C_opt = C[:,ind[:,1]]
    acc_opt = np.trace(C_opt)/np.sum(C_opt)
    print(acc_opt)

    
if __name__ == "__main__":
#    filename = "ATNTFaceImages400.txt"
    filename = "HandWrittenLetters.txt"
    class_ids = [i for i in range(1, 27)]
    test_instances = [-1, -1]
    start(filename, class_ids, test_instances)

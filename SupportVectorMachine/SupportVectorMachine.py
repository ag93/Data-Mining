from sklearn import svm
from sklearn.model_selection import cross_val_score
import numpy as np
import DataHandler as dh
import random


def format_data(filename, class_ids, test_instances):
    data = dh.pickDataClass(filename, class_ids)

    number_per_class = data[0].count(class_ids[0])
    trainX, trainY, testX, testY = dh.splitData2TestTrain(data, number_per_class, test_instances)

    dh.write_2_file(trainX, trainY, testX, testY)


def svm_classifier(train,test):
    X = np.array(train[1:]).transpose().tolist()
    Y = train[0]

    # clf = svm.SVC(decision_function_shape='ovo')
    clf = svm.SVC(kernel='linear', C=1)
    clf.fit(X, Y)

    X_test = np.array(test[1:]).transpose().tolist()
    Y_test = np.array(test[0])

    # count = 0
    # for i in range(len(X_test)):
    #     result = clf.predict([X_test[i]])
    #     if result[0] == Y_test[i]:
    #         count += 1
    # return count / len(Y_test)

    # To compute accuracy using sklearn liabrary
    return clf.score(X_test, Y_test)


# This function is used to perform K-fold cross validation
# k_fold: number of splits for cross-validation, data: data for CV
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
        accuracy = svm_classifier(train, test)
        accuracy_list.append(accuracy)
        print("Iteration", i+1, "accuracy:", accuracy)

    print("Average accuracy:", sum(accuracy_list)/len(accuracy_list))


def start(filename, class_ids, test_instances):
    format_data(filename, class_ids, test_instances)

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

    cross_validation(10 , train)
    '''
    # Use this code for CV using sklearn liabrary
    trainX = np.array(train[1:]).transpose()
    trainY = np.array(train[0]).transpose()
    clf = svm.SVC(kernel='linear', C=1)
    scores = cross_val_score(clf, trainX, trainY, cv=5)
    print(scores)
    '''
    accuracy = svm_classifier(train, test)
    print("\n##################################")
    print("Overall Accuracy:", accuracy)
    return accuracy




if __name__ == "__main__":

    # filename = input("Enter filename: ")
    filename = "HandWrittenLetters.txt"

    # class_ids = [1,2,3,4,5,6,7,8,9,10]
    # class_ids = random.sample(range(1, 27), 5)
    class_ids = [x for x in range(1, 11)]
    print("For class:", class_ids)
    test_instances = [30,38]

    start(filename, class_ids, test_instances)
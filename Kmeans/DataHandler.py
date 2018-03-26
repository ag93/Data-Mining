import string

'''
  filename: char_string specifing the data file to read. For example, 'ATNT_face_image.txt'
  class_ids:  array/list that contains the classes to be pick. For example: (3, 5, 8, 9)
  Returns: an multi-dimension array or a file, containing the data (both attribute vectors and class labels)
           of the selected classes
  We use this subroutine to pick a small part of the data to do experiments. For example for handwrittenletter data,
  we can pick classes "C" and "F" for a 2-class experiment. Or we pick "A,B,C,D,E" for a 5-class experiment.
'''
def pickDataClass(filename, class_ids):
    with open(filename) as f:
        flag = True
        lables = []
        data = []
        for line in f:
            raw_data = line[:-1].split(',')
            if flag:
                lables = raw_data
                flag = False
            row = []

            for i in range(len(raw_data)):
                if int(lables[i]) in class_ids:
                    row.append(int(raw_data[i]))
            data.append(row)

        return data
    
def getLabels(filename, class_ids):
    with open(filename) as f:
        flag = True
        lables = []
        for line in f:
            if flag:
                raw_data = line[:-1].split(',')
                lables = raw_data
                flag = False
        return lables

def splitData2TestTrain(data, number_per_class,  test_instances):
    trainX = []
    testX = []
    trainY = []
    testY = []
    flag = True
    number_of_classes = len(data[0])//number_per_class
    for row in data:
        train_row = []
        test_row = []
        for j in range(number_of_classes):
            for i in range(number_per_class):
                index = j*number_per_class
                if  index+i in range(index + test_instances[0], index + test_instances[1]+1):
                    test_row.append(row[index+i])
                else:
                    train_row.append(row[index+i])

        if flag:
            trainY = train_row
            testY = test_row
            flag = False
        else:
            trainX.append(train_row)
            testX.append(test_row)

    return [trainX, trainY, testX, testY]


def write_2_file(trainX, trainY, testX, testY):
    trainX.insert(0 , trainY)
    testX.insert(0, testY)

    train = ""
    test = ""
    for i in range(len(trainX)):
        string = ""
        for j in trainX[i]:
            string += str(j) + ","
        train += string[:-1] + "\n"

    with open('TrainingData.txt', 'w') as f:
        f.write(train)
    f.close()

    for i in range(len(testX)):
        string = ""
        for j in testX[i]:
            string += str(j) + ","
        test += string[:-1] + "\n"

    with open('TestingData.txt', 'w') as f:
        f.write(test)
    f.close()


def letter_2_digit_convert(letters):
    letters = letters.upper()
    a2z = string.ascii_uppercase

    converted_digits = []

    for letter in letters:
        converted_digits.append(a2z.index(letter) + 1)

    return converted_digits


# This is how you can use the Data Handler to create Train and test dataset and splitting the data.
def run(filename):
    # filename = "trainDataXY.txt"
    class_ids = [1, 3, 5]
    data = pickDataClass(filename, class_ids)
    # print(len(data), data)

    number_per_class = data[0].count(class_ids[0])
    test_instances = [1,3]
    trainX, trainY, testX, testY = splitData2TestTrain(data, number_per_class, test_instances)

    write_2_file(trainX, trainY, testX, testY)

    print(letter_2_digit_convert('EADFBTGH'))


if __name__ == "__main__":
    run("HandWrittenLetters.txt")
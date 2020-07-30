import pickle

import numpy as np


def load_dataset(data_path):
    """Loads the dataset into the different subsets (training, vlaidation, test)

    Arguments:
        data_path {string} -- dataset patch

    Returns:
        array -- different sets
    """
    print('Loading data from file: %s' % (data_path))
    with open(data_path, 'rb') as pickleFile:
        d = pickle.load(pickleFile)
    if len(d) == 6:
        trainX, trainY, valX, valY, testX, testY = d[0], d[1], d[2], d[3], d[4], d[5]
        if len(trainY.shape) > 1:
            trainY = np.argmax(trainY, axis=1)
            valY = np.argmax(valY, axis=1)
            testY = np.argmax(testY, axis=1)
    elif len(d) == 4:
        trainX, trainY, testX, testY = d[0], d[1], d[2], d[3]
        if len(trainY.shape) > 1:
            trainY = np.argmax(trainY, axis=1)
            testY = np.argmax(testY, axis=1)
        trainX, trainY, valX, valY = split_dataset(trainX, trainY)
    else:
        print("No valid dataset")
        exit()
    return trainX, trainY, valX, valY, testX, testY


def split_dataset(set_data, set_labels, split_factor=0.7):
    """Splits the dataset

    Arguments:
        set_data {array} -- data array
        set_labels {array} -- label array

    Keyword Arguments:
        split_factor {float} -- splitting factor for the data (default: {0.7})

    Returns:
        array -- both sets as data and labels
    """
    print("Splitting the data with factor:", str(split_factor))
    set_split = int(np.ceil(len(set_data) * split_factor))

    set_1X = set_data[:set_split, :]
    set_1Y = set_labels[:set_split]

    set_2X = set_data[set_split:, :]
    set_2Y = set_labels[set_split:]

    print('Split set into %s and %s elements' % (len(set_1X), len(set_2X)))
    return set_1X, set_1Y, set_2X, set_2Y


def shuffle_data(trainX, trainY, valX, valY, testX, testY):
    """Shuffles the data

    Arguments:
        trainX {array} -- training data
        trainY {array} -- training labels
        valX {array} -- validation data
        valY {array} -- validation labels
        testX {array} -- test data
        testY {array} -- test labels

    Returns:
        array -- the different sets shuffled
    """
    perm_list = []
    perm = np.random.permutation(len(trainX))
    perm_list.append(perm)
    trainX = trainX[perm]
    trainY = trainY[perm]

    perm = np.random.permutation(len(valX))
    perm_list.append(perm)
    valX = valX[perm]
    valY = valY[perm]

    perm = np.random.permutation(len(testX))
    perm_list.append(perm)
    testX = testX[perm]
    testY = testY[perm]
    return trainX, trainY, valX, valY, testX, testY, perm_list

import os

import numpy as np


def create_mislabel(perc, trainY):
    """Creates the mislabels in the dataset

    Arguments:
        perc {float} -- percentage of mislabels
        trainY {array} -- label data (ground truth)

    Returns:
        array -- mislabeled data, backup labels, mislabeled indices
    """
    classes = set(np.unique(trainY))
    mislabels = int(len(trainY) * perc / 100)
    mis_idx = np.random.permutation(len(trainY))[:mislabels]
    trainY_correct = np.asarray([d for d in trainY])
    for idx in mis_idx:
        false_classes = list(classes.difference(set([trainY[idx]])))
        trainY[idx] = np.random.choice(false_classes)
    print('mis labeled %s examples' % (mislabels))
    return trainY, trainY_correct, mis_idx


def correct_mislabel(trainY, trainY_correct, correction_idx):
    """Compute the corrected mislabels

    Arguments:
        trainY {array} -- label data
        trainY_correct {array} -- correct label data
        correction_idx {array} -- indices to correct

    Returns:
        array -- corrected labels and corrected indices
    """
    corrected = []
    for idx in correction_idx:
        if trainY[idx] != trainY_correct[idx]:
            corrected.append(idx)
            trainY[idx] = trainY_correct[idx]
    print('Corrected %s examples by inspection' % (len(corrected)))
    return trainY, corrected


def get_sorted_corrections(options, trainY):
    """Prvides the lsit of indices to inspect to correct the data according to the approach ranking

    Arguments:
        options {dict} -- parameter dictionary
        trainY {array} -- label array

    Returns:
        array -- indices sorted by ranking
    """
    # mode desicdes which ranking approach is used
    if options.correction_mode == 0:
        if options.each_class:
            print('Influence class based correction')
            sorted_indices = get_class_based_influence(options, trainY)
        else:
            print('Influence based correction')
            scores = np.load(options.influence_file, allow_pickle=True)
            if options.absolute_influence:
                scores = abs(scores)
            sorted_indices = np.argsort(scores)
    elif options.correction_mode == 1:
        print('Loss based correction')
        scores = np.load(options.loss_file, allow_pickle=True)
        sorted_indices = np.argsort(scores)
    elif options.correction_mode == 2:
        print('Representer based correction')
        scores, _ = np.load(options.representer_path +
                            'representer_influence.npy', allow_pickle=True)
        #sorted_indices = np.argsort(np.max(scores, axis=1))
        sorted_indices = np.argsort(np.max(abs(scores), axis=1))
    else:
        print('Random correction')
        sorted_indices = np.random.permutation(len(trainY))
    return sorted_indices


def get_class_based_influence(options, trainY):
    """Gets the class based influence scores

    Arguments:
        options {dict} -- parameter dictionary
        trainY {array} -- label array

    Returns:
        array -- claswise computed and concated score
    """
    concat_scores = np.zeros(len(trainY))
    file_found = False
    # get the influence for each class
    for filename in os.listdir(options.influence_path):
        if 'class'in filename:
            file_found = True
            c = int(filename.split('class_')[-1].split('_')[-1].split('.')[0])
            print('Class: %s | Influence scores from: %s' % (c, filename))
            scores = np.load(os.path.join(options.influence_path, filename), allow_pickle=True)
            if options.absolute_influence:
                scores = abs(scores)
            concat_scores = np.add(concat_scores, scores)
    indices_order = np.argsort(concat_scores)
    scores = np.sort(concat_scores)
    if len(concat_scores) == 0:
        if not file_found:
            print('No valid class files')
        else:
            print('No need to retrain cause no mislabels were found')
        exit()
    indices_order = np.argsort(concat_scores)
    return np.asarray(indices_order)


def create_correction_set(options, trainY):
    """Returns the indices to inspect

    Arguments:
        options {dict} -- parameter dictionary
        trainY {array} -- label array

    Returns:
        array -- indices selected for inspection
    """
    sorted_indices = get_sorted_corrections(options, trainY)

    m_correction_number = int(len(trainY) * options.manual_correction / 100)
    if not options.correct_low:
        harmful = set(sorted_indices[::-1][:m_correction_number])
    else:
        harmful = set(sorted_indices[:m_correction_number])
    return harmful


def remove_importance(options, trainX, trainY, mis_idx):
    """Removes samples instead of inspecting them

    Arguments:
        options {dict} -- parameter dictionary
        trainX {array} -- data array
        trainY {array} -- label array
        mis_idx {array} -- ranking scores

    Returns:
        array -- filtered data, labels, removed data and mislabled removed
    """
    sorted_indices = get_sorted_corrections(options, trainY)

    if options.remove_high:
        sorted_indices = sorted_indices[::-1]

    skip = int(len(trainY) * options.remove_perc / 100)
    remove = set(sorted_indices[:skip])

    mislabelled = []
    label_check = False
    if mis_idx is not None:
        label_check = True
        mis_set = set(mis_idx)
    filteredX = []
    filteredY = []
    for i in range(len(trainY)):
        if i in remove:
            if label_check:
                if i in mis_set:
                    mislabelled.append(i)
            continue
        filteredX.append(trainX[i])
        filteredY.append(trainY[i])
    if not options.remove_high:
        print('Removed the %s low score examples' % (skip))
    else:
        print('Removed the %s high score examples' % (skip))
    if label_check:
        print('Removed includes %s mis labels' % (len(mislabelled)))
    return np.asarray(filteredX), np.asarray(filteredY), remove, mislabelled

import darkon
import matplotlib as plt
import numpy as np
import tensorflow as tf


class MyFeeder(darkon.InfluenceFeeder):
    """Influence function sample feeder

    Arguments:
        darkon {obj} -- influence feeder
    """

    def __init__(self, trainX, trainY, testX, testY):
        """Initialize the variables

        Arguments:
            trainX {array} -- train data
            trainY {array} -- train labels
            testX {array} -- test data
            testY {array} -- test labels
        """
        self.train_label = trainY
        self.train_data = trainX

        self.test_label = testY
        self.test_data = testX

        self.train_batch_offset = 0

    def test_indices(self, indices):
        """Return the test data and labels

        Arguments:
            indices {array} -- indices to use

        Returns:
            array -- data nd labels for the indices
        """
        return self.test_data[indices], self.test_label[indices]

    def train_batch(self, batch_size):
        """Retruns the training batch for the influence method

        Arguments:
            batch_size {int} -- batch size

        Returns:
            array -- data and label batch
        """
        # calculate offset
        start = self.train_batch_offset
        end = start + batch_size
        self.train_batch_offset += batch_size
        return self.train_data[start:end, ...], self.train_label[start:end, ...]

    def train_one(self, idx):
        """Returns a single sample

        Arguments:
            idx {int} -- index used to select the data

        Returns:
            array -- data and label sample
        """
        return self.train_data[idx, ...], self.train_label[idx, ...]

    def reset(self):
        """Resets the batch index
        """
        self.train_batch_offset = 0


def set_up_influence_feeder(options, net, trainX, trainY, valX, valY, testX, testY):
    """Creates the influence feeder for a given network and the data

    Arguments:
        options {dict} -- parameter dictionary
        net {model} -- inference network
        trainX {array} -- training data
        trainY {array} -- training labels
        valX {array} -- validation data
        valY {array} -- validation labels
        testX {array} -- test data
        testY {array} -- test labels

    Returns:
        array -- the influence feeder the set inspector and the session
    """
    # data feeder
    if options.influence_over_set == 0:
        print('Influence over train set')
        feeder = MyFeeder(trainX, trainY, trainX, trainY)
    elif options. influence_over_set == 1:
        print('Influence over val set')
        feeder = MyFeeder(trainX, trainY, valX, valY)
    else:
        print('Influence over test set')
        feeder = MyFeeder(trainX, trainY, testX, testY)

    # network
    check_point = options.model_path

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess, check_point)

    trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    # influence inspector
    inspector = darkon.Influence(
        workspace=options.influence_path,
        feeder=feeder,
        loss_op_train=net.full_loss,
        loss_op_test=net.loss_op,
        x_placeholder=net.series_placeholder,
        y_placeholder=net.label_placeholder,
        trainable_variables=trainable_variables)
    return feeder, inspector, sess


def compute_influence_each(options, feeder, inspector, sess):
    """Computes the influence for each class

    Arguments:
        options {dict} -- parameter dictionary
        feeder {obj} -- influence feeder
        inspector {obj} -- influence inspector
        sess {session} -- session used to compute
    """
    classes = np.unique(feeder.test_label)
    for c in classes:
        print('Compute Scores for class', int(c))
        options.only_class = int(c)
        compute_influences(options, feeder, inspector, sess)


def compute_influence_example(options, feeder, inspector, sess):
    """Computes the influence of given samples

    Arguments:
        options {dict} -- parameter dictionary
        feeder {obj} -- influence feeder
        inspector {obj} -- influence inspector
        sess {session} -- session used to compute

    Returns:
        array -- influence scores
    """
    influence_target = options.example_idx

    # if a single sample is passed it is visualized
    if len(options.example_idx) == 1 and not options.compute_all_examples:
        title_str = 'Label:' + str(feeder.test_label[influence_target[0]])
        plt.title(title_str)
        plt.plot(feeder.test_data[influence_target[0]])

    scores = compute_influences(options, feeder, inspector, sess)

    return scores


def compute_influences(options, feeder, inspector, sess):
    """Computes the influence scores

    Arguments:
        options {dict} -- parameter dictionary
        feeder {obj} -- influence feeder
        inspector {obj} -- influence inspector
        sess {session} -- session used to compute

    Returns:
        array -- influence scores
    """
    # compute mode
    if options.compute_all_examples:
        influence_target = np.arange(len(feeder.test_data))
        if options.equalize:
            _, influence_target = equal_per_class(feeder.test_label)
        if options.only_class > -1:
            influence_target = select_class(
                feeder.test_label, options.only_class)
    else:
        influence_target = options.example_idx

    test_indices = influence_target
    testset_batch_size = options.batch_size_test

    train_batch_size = options.batch_size
    if options.influence_iterations > 0:
        train_iterations = options.influence_iterations
    else:
        train_iterations = int(
            np.ceil(len(feeder.train_data) / train_batch_size))

    hess_iter = int(np.ceil(options.hessian_depth *
                            options.hessian_batch_size))
    hess_max_iter = int(
        np.ceil(len(feeder.train_data) / options.hessian_batch_size))
    if hess_iter > train_iterations:
        options.hessian_depth = hess_max_iter

    approx_params = {
        'scale': options.hessian_scale,
        'damping': options.hessian_dumping,
        'num_repeats': options.hessian_repeats,
        'recursion_depth': options.hessian_depth,
        'recursion_batch_size': options.hessian_batch_size
    }

    scores = inspector.upweighting_influence_batch(
        sess=sess,
        test_indices=test_indices,
        test_batch_size=testset_batch_size,
        approx_params=approx_params,
        train_batch_size=train_batch_size,
        train_iterations=train_iterations,
        subsamples=options.subsamples,
        force_refresh=options.refresh)

    scores = scores[:len(feeder.train_data)]

    # save options including classwise and samplewise
    if options.save_influences:
        if options.compute_all_examples:
            if options.equalize:
                ipath = options.influence_path + "/influence_scores_equal.npy"
            else:
                ipath = options.influence_path + "/influence_scores.npy"
            if options.only_class > -1:
                ipath = options.influence_path + "/influence_scores_class_" + \
                    str(options.only_class) + ".npy"

        else:
            idx_str = ""
            for i in influence_target:
                idx_str = idx_str + "_" + str(i)
            ipath = options.influence_path + "/influence_scores" + idx_str + ".npy"
        np.save(ipath, scores)

    return scores


def equal_per_class(dataY):
    """Equalizes the number of samples per class for a balanced score

    Arguments:
        dataY {array} -- labels

    Returns:
        array -- number per class and equalized label index array
    """
    uniques, counts = np.unique(dataY, return_counts=True)
    per_class = np.min(counts)
    class_counts = np.zeros(len(uniques))
    total_counts = 0
    max_count = len(uniques) * per_class
    idx = 0
    equal_indices = []
    while total_counts < max_count:
        c = int(dataY[idx])
        if class_counts[c] < per_class:
            equal_indices.append(idx)
            class_counts[c] += 1
            total_counts += 1
        idx += 1
    print('Selected %s samples per class' % (per_class))
    return per_class, np.asarray(equal_indices)


def select_class(dataY, class_idx):
    """Selects a given class and their indices

    Arguments:
        dataY {array} -- label array
        class_idx {int} -- class to select

    Returns:
        array -- indices of the given class
    """
    int_label = np.asarray([int(d) for d in dataY])
    idx_list = np.squeeze(np.argwhere(int_label == class_idx))
    print('Selected %s samples class' % (len(idx_list)))
    return idx_list


def show_most_influencing(dataX, dataY, scores, showX):
    """Plots the relevant samples

    Arguments:
        dataX {array} -- data array
        dataY {array} -- label array
        scores {array} -- influence scores
        showX {int} -- number of samples to show
    """
    sorted_indices = np.argsort(scores)
    showX = np.min([showX, len(sorted_indices)])
    harmful = sorted_indices[:showX]
    helpful = sorted_indices[-showX:][::-1]

    print('\nHelpful:')
    for pos, idx in enumerate(helpful):
        print('Position: %s | Idx: %s | Score: %s | Class: %s' %
              (pos, idx, scores[idx], int(dataY[idx])))

    print('\nHarmful:')
    for pos, idx in enumerate(harmful):
        print('Position: %s | Idx: %s | Score: %s | Class: %s' %
              (pos, idx, scores[idx], int(dataY[idx])))

    plot_size = int(np.ceil(np.sqrt(showX)))

    fig, axes1 = plt.subplots(plot_size, plot_size, figsize=(15, 10))
    fig.suptitle("Helpful")
    target_idx = 0
    for j in range(plot_size):
        for k in range(plot_size):
            if target_idx < showX:
                idx = helpful[target_idx]
                axes1[j][k].set_axis_off()
                axes1[j][k].plot(dataX[idx])
                axes1[j][k].set_title('Idx: %s | Class: %s' %
                                      (idx, int(dataY[idx])))
            else:
                axes1[j][k].set_visible(False)

            target_idx += 1

    fig, axes1 = plt.subplots(plot_size, plot_size, figsize=(15, 10))
    fig.suptitle("Harmful")
    target_idx = 0
    for j in range(plot_size):
        for k in range(plot_size):
            if target_idx < showX:
                idx = harmful[target_idx]
                axes1[j][k].set_axis_off()
                axes1[j][k].plot(dataX[idx])
                axes1[j][k].set_title('Idx: %s | Class: %s' %
                                      (idx, int(dataY[idx])))
            else:
                axes1[j][k].set_visible(False)

            target_idx += 1

    plt.show()

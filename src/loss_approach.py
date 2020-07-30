import numpy as np


def get_train_sample_losses(net, options, trainX, trainY):
    """Compute the loss per sample

    Arguments:
        net {model} -- network for inference
        options {dict} -- parameter dict
        trainX {array} -- training data
        trainY {array} -- training labels
    """
    _, _, loss_per_sample, _ = net.test(options, trainX, trainY)
    np.save(options.loss_file, np.concatenate(loss_per_sample))

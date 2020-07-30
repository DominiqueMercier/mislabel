import os
from optparse import OptionParser

import numpy as np
import tensorflow as tf

from dataset import load_dataset, shuffle_data
from influence_approach import (compute_influence_each,
                                compute_influence_example,
                                set_up_influence_feeder, show_most_influencing)
from loss_approach import get_train_sample_losses
from mislabel import (correct_mislabel, create_correction_set, create_mislabel,
                      remove_importance)
from model_trainer import Train, compare_labels, perform_testing
from representer_approach import perform_representer_influence

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

def process(options):
    # fix the random seed for reproducability
    np.random.seed(options.seed)

    # load the data
    trainX, trainY, valX, valY, testX, testY = load_dataset(options.data_path)
    trainX = np.transpose(trainX, [0, 2, 1])
    valX = np.transpose(valX, [0, 2, 1])
    testX = np.transpose(testX, [0, 2, 1])

    # get meta data
    if options.get_data_shapes:
        options.timesteps = trainX.shape[2]
        options.channels = trainX.shape[1]
        options.classes = len(np.unique(trainY))
        print('Found %s timesteps with %s channel(s) and %s classes' %
              (options.timesteps, options.channels, options.classes))

    # shuffle the data
    if options.shuffle:
        trainX, trainY, valX, valY, testX, testY, perm_list = shuffle_data(
            trainX, trainY, valX, valY, testX, testY)
        np.save(options.train_dir + "/shuffle.npy", perm_list)

    # shrink data
    if options.shrink_data < 100:
        train_shrink = int(len(trainY) * options.shrink_data / 100)
        val_shrink = int(len(valY) * options.shrink_data / 100)
        test_shrink = int(len(testY) * options.shrink_data / 100)

        trainX = trainX[:train_shrink]
        valX = valX[:val_shrink]
        testX = testX[:test_shrink]
        trainY = trainY[:train_shrink]
        valY = valY[:val_shrink]
        testY = testY[:test_shrink]

    # create mislabels
    if options.mislabel:
        trainY, trainY_correct, mis_idx = create_mislabel(
            options.mislabel_perc, trainY)
        np.save(options.train_dir + "/mislabel.npy", mis_idx)

    # create mislabels for validation
    if options.mislabel_val:
        valY, valY_correct, mis_idx = create_mislabel(
            options.mislabel_perc, valY)
        np.save(options.train_dir + "/mislabel_val.npy", mis_idx)

    # remove based on importance value
    if options.remove_low or options.remove_high:
        if options.mislabel:
            trainX, trainY, remove_list, corrected = remove_importance(
                options, trainX, trainY, mis_idx)
            np.save(options.train_dir + "/removed_correction.npy", corrected)
        else:
            trainX, trainY, remove_list, _ = remove_importance(
                options, trainX, trainY, None)
        np.save(options.train_dir + "/removement_list.npy", remove_list)

    # manual correct data
    if options.manual_correction > 0 and options.mislabel:
        correction_list = create_correction_set(options, trainY)
        np.save(options.train_dir + "/correction_list.npy", correction_list)
        trainY, corrected = correct_mislabel(
            trainY, trainY_correct, correction_list)
        np.save(options.train_dir + "/manual_correction.npy", corrected)

    print('Train set:', str(trainX.shape))
    print('Val set:', str(valX.shape))
    print('Test set:', str(testX.shape))
    print('Classes:', str(options.classes))
    
    # Initialize the Train object
    net = Train(options.timesteps, options.channels, options.classes)
    net.build_graph()
    # Start the training session
    if options.train:
        net.train(options, trainX, trainY, valX, valY)
    # testing
    if options.test:
        perform_testing(options, net, trainX, trainY, valX, valY, testX, testY)
    # collect train loss
    if options.collect_train_loss:
        get_train_sample_losses(net, options, trainX, trainY)
    # influences calc
    if options.influences:
        influence_feeder, inspector, sess = set_up_influence_feeder(
            options, net, trainX, trainY, valX, valY, testX, testY)
        if options.compute_score:
            if options.each_class:
                compute_influence_each(
                    options, influence_feeder, inspector, sess)
            else:
                influence_scores = compute_influence_example(
                    options, influence_feeder, inspector, sess)
                if options.show_influence:
                    show_most_influencing(
                        influence_feeder.train_data, influence_feeder.train_label, influence_scores, options.show_number)
    # influence show
    if options.show_influence and not options.each_class:
        influence_scores = np.load(options.influence_file, allow_pickle=True)
        show_most_influencing(
            trainX, trainY, influence_scores, options.show_number)
    # representer influence
    if options.compute_representer_influence:
        net.test(options, trainX, trainY)
        perform_representer_influence(options)
    # compare labels
    if options.predict:
        setX = trainX
        setY = trainY
        if options.predict_set == 1:
            setX = valX
            setY = valY
        if options.predict_set == 2:
            setX = testX
            setY = testY
        prediction_array, _, _,  acc = net.test(
            options, setX[options.predict_start:options.predict_end], setY[options.predict_start:options.predict_end])
        compare_labels(
            setY[options.predict_start:options.predict_end], prediction_array)
        print('Final accuracy: %s' % acc)


if __name__ == "__main__":
    # Command line options
    parser = OptionParser()

    # Base options
    parser.add_option("--get_data_shapes", action="store_true",
                      dest="get_data_shapes", default=True, help="get data shapes")
    parser.add_option("--timesteps", action="store", type="int",
                      dest="timesteps", default=50, help="timesteps")
    parser.add_option("--channels", action="store", type="int",
                      dest="channels", default=3, help="channel")
    parser.add_option("--classes", action="store", type="int",
                      dest="classes", default=2, help="classes")
    parser.add_option("--data_path", action="store", type="string", dest="data_path",
                      default='../datasets/anomaly_data.pickle', help="Data path")

    parser.add_option("--shuffle", action="store_true",
                      dest="shuffle", default=True, help="shuffle")
    parser.add_option("--seed", action="store", type="int",
                      dest="seed", default=0, help="random seed")

    parser.add_option("--shrink_data", action="store", type="float",
                      dest="shrink_data", default=100, help="select percentage of data")

    parser.add_option("--model_path", action="store", type="string",
                      dest="model_path", default="", help="Model path")
    parser.add_option("--restore", action="store_true",
                      dest="restore", default=False, help="restore")
    parser.add_option("--restore_best", action="store_true",
                      dest="restore_best", default=False, help="restore best model")
    parser.add_option("--train_dir", action="store", type="string",
                      dest="train_dir", default="../models/model", help="train path")

    parser.add_option("--batch_size", action="store", type="int",
                      dest="batch_size", default=32, help="Batch size")
    parser.add_option("--batch_size_test", action="store", type="int",
                      dest="batch_size_test", default=32, help="Batch size test")
    parser.add_option("--epochs", action="store", type="int",
                      dest="epochs", default=50, help="epochs")
    parser.add_option("--report_size", action="store", type="int",
                      dest="report_size", default=100, help="report after x steps")

    parser.add_option("--learning_rate", action="store", type="float",
                      dest="learning_rate", default=0.001, help="learning rate")
    parser.add_option("--lr_patience", action="store", type="int",
                      dest="lr_patience", default=4, help="lr patience")
    parser.add_option("--early_patience", action="store", type="int",
                      dest="early_patience", default=10, help="early stopping")

    parser.add_option("--train", action="store_true",
                      dest="train", default=False, help="training")
    parser.add_option("--test", action="store_true",
                      dest="test", default=False, help="test")
    parser.add_option("--test_set", action="store", type="int", dest="test_set",
                      default=3, help="set used for test (train, val, test, all")
    parser.add_option("--influences", action="store_true",
                      dest="influences", default=False, help="influences")
    parser.add_option("--collect_train_loss", action="store_true",
                      dest="collect_train_loss", default=False, help="collect the loss per sample")

    parser.add_option("--influence_path", action="store", type="string",
                      dest="influence_path", default="", help="influence path")
    parser.add_option("--compute_score", action="store_true",
                      dest="compute_score", default=False, help="compute influence example")
    parser.add_option("--example_idx", action="store", type="string",
                      dest="example_idx", default="0", help="example number")

    parser.add_option("--influence_iterations", action="store", type="int",
                      dest="influence_iterations", default=-1, help="influence iterations")
    parser.add_option("--hessian_scale", action="store", type="int",
                      dest="hessian_scale", default=1e4, help="hessian scale")
    parser.add_option("--hessian_dumping", action="store", type="float",
                      dest="hessian_dumping", default=0.01, help="hessian dumping")
    parser.add_option("--hessian_repeats", action="store", type="int",
                      dest="hessian_repeats", default=1, help="hessian repeats")
    parser.add_option("--hessian_batch_size", action="store", type="int",
                      dest="hessian_batch_size", default=32, help="hessian batch size")
    parser.add_option("--hessian_depth", action="store", type="int",
                      dest="hessian_depth", default=10000, help="hessian depth")

    parser.add_option("--compute_all_examples", action="store_true",
                      dest="compute_all_examples", default=False, help="compute influence for all examples")
    parser.add_option("--influence_over_set", action="store", type="int",
                      dest="influence_over_set", default=0, help="compute influence over train, val or test")
    parser.add_option("--save_influences", action="store_true",
                      dest="save_influences", default=True, help="save influences")

    parser.add_option("--subsamples", action="store", type="int",
                      dest="subsamples", default=-1, help="subsample for influence")
    parser.add_option("--refresh", action="store_true", dest="refresh",
                      default=False, help="refresh the influence results")

    parser.add_option("--mislabel", action="store_true",
                      dest="mislabel", default=False, help="mislabels train data")
    parser.add_option("--mislabel_val", action="store_true",
                      dest="mislabel_val", default=False, help="mislabels val data")
    parser.add_option("--mislabel_perc", action="store", type="float",
                      dest="mislabel_perc", default=10, help="mislabel percentage")
    parser.add_option("--correct_low", action="store_true",
                      dest="correct_low", default=False, help="correct low score samples")

    parser.add_option("--equalize", action="store_true", dest="equalize",
                      default=False, help="equalize test labels to same number")
    parser.add_option("--only_class", action="store", type="int",
                      dest="only_class", default=-1, help="influence for class x")
    parser.add_option("--each_class", action="store_true",
                      dest="each_class", default=False, help="influence for each class")
    parser.add_option("--absolute_influence", action="store_true",
                      dest="absolute_influence", default=False, help="most influential samples")

    parser.add_option("--show_influence", action="store_true",
                      dest="show_influence", default=False, help="show influence plots")
    parser.add_option("--show_number", action="store", type="int",
                      dest="show_number", default=10, help="show top x results")
    parser.add_option("--influence_file", action="store", type="string",
                      dest="influence_file", default="", help="path to the influence file")
    parser.add_option("--loss_file", action="store", type="string",
                      dest="loss_file", default="", help="path to the loss file")

    parser.add_option("--manual_correction", action="store", type="int",
                      dest="manual_correction", default=0, help="percentage of data to correct manual")
    parser.add_option("--correction_mode", action="store", type="int",
                      dest="correction_mode", default=0, help="influence, loss, representer, random")
    parser.add_option("--remove_low", action="store_true",
                      dest="remove_low", default=False, help="removes harmfull samples")
    parser.add_option("--remove_high", action="store_true",
                      dest="remove_high", default=False, help="removes helpfull samples")
    parser.add_option("--remove_perc", action="store", type="float",
                      dest="remove_perc", default=10, help="removes harmful percentage")

    parser.add_option("--save_representer_dataset", action="store_true",
                      dest="save_representer_dataset", default=False, help="computes the representer dataset")
    parser.add_option("--representer_path", action="store", type="string",
                      dest="representer_path", default="", help="representer data path")

    parser.add_option("--compute_representer_influence", action="store_true",
                      dest="compute_representer_influence", default=False, help="compute the representer influence")

    parser.add_option("--predict", action="store_true",
                      dest="predict", default=False, help="predict samples")
    parser.add_option("--predict_set", action="store", type="int",
                      dest="predict_set", default=0, help="predict set train, val, test")
    parser.add_option("--predict_start", action="store", type="int",
                      dest="predict_start", default=0, help="start idx")
    parser.add_option("--predict_end", action="store", type="int",
                      dest="predict_end", default=10, help="end idx")

    # Parse command line options
    (options, args) = parser.parse_args()

    # adjust parameter
    if options.model_path == "":
        options.model_path = options.train_dir

    if options.influence_path == "":
        options.influence_path = options.train_dir + "/influence-workspace"

    if options.influence_file == "":
        options.influence_file = options.influence_path + "/influence_scores.npy"

    if options.loss_file == "":
        options.loss_file = options.train_dir + "/loss.npy"

    if options.restore_best:
        options.restore = True
        latest_model = tf.train.latest_checkpoint(options.train_dir)
        options.model_path = latest_model
        print('Model selected:', latest_model)

    if options.representer_path == "":
        options.representer_path = options.train_dir + "/representer_data/"

    # print options
    print(options)

    # Convert the string to a list if indices
    options.example_idx = np.fromstring(
        options.example_idx, dtype=int, sep=',')

    if not os.path.exists(options.train_dir):
        os.makedirs(options.train_dir)

    if not os.path.exists(options.representer_path):
        os.mkdir(options.representer_path)

    process(options)

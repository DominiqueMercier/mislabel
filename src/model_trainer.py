import os
import sys
import time

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from model import inference


class Train():
    """Trainer class to perform the training, testing and initialization of the model
    """

    def __init__(self, timesteps, channels, classes):
        """Initialize the model

        Arguments:
            timesteps {int} -- number of timesteps
            channels {int} -- number of channels
            classes {int} -- number of classes
        """
        self.timesteps = timesteps
        self.channels = channels
        self.classes = classes
        self.placeholders()

    def placeholders(self):
        """Create the placeholders for the model
        """
        self.series_placeholder = tf.placeholder(
            dtype=tf.float32, shape=[None, self.channels, self.timesteps])
        self.label_placeholder = tf.placeholder(dtype=tf.int32, shape=[None])

        self.lr_placeholder = tf.placeholder(dtype=tf.float32, shape=[])

    def build_graph(self):
        """Builds the graph for the model
        """
        global_step = tf.Variable(0, trainable=False)

        layers = inference(self.series_placeholder, self.classes)
        logits = layers[-1]
        
        self.layers = layers

        loss, loss_per_sample = self.loss(logits, self.label_placeholder)
        self.full_loss = tf.add_n([loss])
        self.loss_op = loss
        self.loss_per_sample = loss_per_sample

        predictions = tf.nn.softmax(logits)
        self.predictions_op = predictions

        acc = self.accuracy_operation()
        self.full_acc = tf.reduce_mean([acc])
        self.accuracy_op = acc
        
        self.train_op = self.train_operation(
            global_step, self.full_loss, self.full_acc)

    def generate_batch(self, data, label, batch_size, batch_id):
        """Creates batches of data

        Arguments:
            data {array} -- data array
            label {array} -- label array
            batch_size {nit} -- size of a batch
            batch_id {int} -- batch id used to create subset

        Returns:
            array -- data array and label array for the given batch id
        """
        start = batch_size * batch_id
        end = np.min([len(data), start + batch_size])
        data_batch = data[start:end, ...]
        label_batch = label[start:end]
        return data_batch, label_batch

    def train(self, options, trainX, trainY, valX, valY):
        """Trains the model with the given dataset

        Arguments:
            options {dict} -- training options as dictionary
            trainX {array} -- training data
            trainY {array} -- training labels
            valX {array} -- validation data
            valY {array} -- validation labels
        """
        saver = tf.train.Saver(tf.global_variables())
        summary_op = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        # load already saved model
        if options.restore is True:
            saver.restore(sess, options.model_path)
            print('\nRestored from checkpoint... {}'.format(options.model_path))
        else:
            sess.run(init)

        summary_writer = tf.summary.FileWriter(options.train_dir, sess.graph)

        last_loss = sys.float_info.max
        last_acc = 0
        lr_patience = 0
        early_patience = 0

        lr = options.learning_rate

        print('Start training...')
        print('--------------------------------------------------')

        # loop over epochs
        for epoch in range(options.epochs):
            print('Epoch: %s ' % (epoch+1))
            start_time = time.time()

            train_loss = []
            train_acc = []
            max_batch = int(np.ceil(len(trainX) / options.batch_size))

            # process each batch
            for batchidx in range(max_batch):
                step = epoch * max_batch + batchidx

                train_batch_data, train_batch_labels = self.generate_batch(
                    trainX, trainY, options.batch_size, batchidx)

                summary_str, _, train_loss_value, train_acc_value = sess.run([summary_op, self.train_op, self.full_loss, self.accuracy_op],
                                                                             {self.series_placeholder: train_batch_data,
                                                                              self.label_placeholder: train_batch_labels,
                                                                              self.lr_placeholder: lr})
                train_loss.append(train_loss_value)
                train_acc.append(train_acc_value)

                if (step - epoch*max_batch) % options.report_size == 0:
                    summary_writer.add_summary(summary_str, step)

                    tqdm.write('Step: %s / %s | Loss %s | Acc: %s' % (step+1 - epoch *
                                                                      max_batch, max_batch, np.average(train_loss), np.average(train_acc)))

            duration = time.time() - start_time
            print('Time spent: %s seconds' % (duration))
            train_loss = np.average(train_loss)
            train_acc = np.average(train_acc)
            print('Epoch: %s | Train loss: %s | Train acc: %s' %
                  (epoch+1, train_loss, train_acc))

            # compute val loss
            val_loss = []
            val_acc = []
            max_val_batch = int(np.ceil(len(valX) / options.batch_size))

            # perform validation for each batch
            for batchidx in range(max_val_batch):
                validation_batch_data, validation_batch_labels = self.generate_batch(
                    valX, valY, options.batch_size, batchidx)

                validation_loss_value, val_acc_value = sess.run([self.full_loss, self.accuracy_op],
                                                                {self.series_placeholder: validation_batch_data,
                                                                 self.label_placeholder: validation_batch_labels,
                                                                 self.lr_placeholder: lr})
                val_loss.append(validation_loss_value)
                val_acc.append(val_acc_value)

                if step % options.report_size == 0:
                    tqdm.write('Step: %s / %s | Loss %s | Acc: %s' % (step+1,
                                                                      max_batch, np.average(val_loss), np.average(val_acc)))

            val_loss = np.average(val_loss)
            val_acc = np.average(val_acc)
            print('Epoch: %s | Validation loss: %s | Validation acc: %s' %
                  (epoch+1, val_loss, val_acc))
            print('--------------------------------------------------')

            if val_acc > last_acc:
                last_acc = val_acc

            if val_loss < last_loss:
                lr_patience = 0
                early_patience = 0
                last_loss = val_loss

                checkpoint_path = os.path.join(options.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
            else:
                lr_patience += 1
                early_patience += 1

            if lr_patience == options.lr_patience:
                lr = 0.5 * lr
                print('Learning rate decayed to ', str(lr))

            if early_patience == options.early_patience:
                print('Early stopped at epoch:', str(epoch+1))
                break
        print('Final Validation acc: %s' % (last_acc))

    def test(self, options, testX, testY):
        """Test the model and create feature datasets

        Arguments:
            options {dict} -- parameter dictionary
            testX {array} -- test data
            testY {array} -- test labels

        Returns:
            array -- holds the predictions, loss, loss per sample and the accuracy
        """
        num_test_series = len(testX)
        num_batches = int(np.ceil(num_test_series / options.batch_size_test))
        print('%i test batches in total...' % num_batches)

        # Initialize a new session and restore a checkpoint
        saver = tf.train.Saver(tf.global_variables())
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        
        if not os.path.isfile(options.model_path):
            latest_model = tf.train.latest_checkpoint(options.train_dir)
            options.model_path = latest_model

        saver.restore(sess, options.model_path)
        print('Model restored from ', options.model_path)
               
        # save the weights of the last layer
        if options.save_representer_dataset:
            #print(tf.all_variables())
            w_t, b_t = [t for t in  tf.all_variables() if 'dense3/dense_3/kernel:0' in t.name or 'dense3/dense_3/bias:0' in t.name]
            w, b = sess.run([w_t, b_t])
            print('Kernel shape:', w.shape)
            print('Bias shape:', b.shape)
            
            np.save(options.representer_path +
                    'model_last_weights.npy', [w, b])

            feature_array = None

        prediction_array = None

        loss = []
        loss_per_sample = []
        acc = []
        # iterate through the test batches
        for step in range(num_batches):
            test_batch_data, test_batch_labels = self.generate_batch(
                testX, testY, options.batch_size_test, step)

            # exclude the features if not used for representer
            if not options.save_representer_dataset:
                batch_prediction_array, test_loss, test_loss_per_sample, test_acc = sess.run([self.predictions_op, self.full_loss, self.loss_per_sample, self.accuracy_op],
                                                                                             feed_dict={self.series_placeholder: test_batch_data,
                                                                                                        self.label_placeholder: test_batch_labels})
            else:
                batch_feature_array, batch_prediction_array, test_loss, test_loss_per_sample, test_acc = sess.run([self.layers[-3], self.predictions_op, self.full_loss, self.loss_per_sample, self.accuracy_op],
                                                                                                                  feed_dict={self.series_placeholder: test_batch_data,
                                                                                                                             self.label_placeholder: test_batch_labels})
                if not feature_array is None:
                    feature_array = np.concatenate(
                        (feature_array, batch_feature_array))
                else:
                    feature_array = batch_feature_array

            loss.append(test_loss)
            loss_per_sample.append(test_loss_per_sample)
            acc.append(test_acc)
            if not prediction_array is None:
                prediction_array = np.concatenate(
                    (prediction_array, batch_prediction_array))
            else:
                prediction_array = batch_prediction_array

            if step % options.report_size == 0:
                tqdm.write('Step: %s / %s | Loss: %s | Acc: %s' %
                           (step+1, num_batches, np.average(loss), np.average(acc)))

        loss = np.average(loss)
        acc = np.average(acc)
        print('Test accuracy: %s' % (acc))

        # save feature and label dataset for representer
        if options.save_representer_dataset:
            np.save(options.representer_path +
                    'feature_dataset.npy', feature_array)
            np.save(options.representer_path +
                    'label_dataset.npy', prediction_array)

        return prediction_array, loss, loss_per_sample, acc

    def loss(self, logits, labels):
        """Compute the CE loss

        Arguments:
            logits {array} -- logits after computation
            labels {arrray} -- ground truth labels

        Returns:
            array -- CE mean and samplewise CE
        """
        num_classes = logits.shape[-1]
        labels = tf.one_hot(labels, num_classes, axis=-1)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                                   labels=labels,
                                                                   name='cross_entropy_per_example')

        cross_entropy_mean = tf.reduce_mean(
            cross_entropy, name='cross_entropy')
        return cross_entropy_mean, cross_entropy

    def train_operation(self, global_step, total_loss, total_acc):
        """Training operation to minimize the objective

        Arguments:
            global_step {int} -- global step during training
            total_loss {float} -- model loss
            total_acc {float} -- model accuarcy

        Returns:
            obj -- train step for the model
        """
        # Add train_loss, current learning rate and train error into the tensorboard summary ops
        tf.summary.scalar('learning_rate', self.lr_placeholder)
        tf.summary.scalar('train_loss', total_loss)
        tf.summary.scalar('train_acc', total_acc)

        opt = tf.train.AdamOptimizer(learning_rate=self.lr_placeholder)
        train_op = opt.minimize(total_loss, global_step=global_step)
        return train_op

    def accuracy_operation(self):
        """Computes the accuarcy

        Returns:
            float -- accuracy for the current input
        """
        correct_pred = tf.equal(tf.argmax(self.predictions_op, 1), tf.cast(
            self.label_placeholder, tf.int64))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        return accuracy


def perform_testing(options, net, trainX, trainY, valX, valY, testX, testY):
    """Computes the statistics for all dataset (training, validation, testing)

    Arguments:
        options {dict} -- parameter dictionary
        net {model} -- the inference model
        trainX {array} -- training data
        trainY {array} -- training labels
        valX {array} -- validation data
        valY {array} -- validation labels
        testX {array} -- test data
        testY {array} -- test labels
    """
    test_loss = []
    test_acc = []
    sets = []
    root_representer = options.representer_path
    # checks which parameter ist passed as test set and includes the corresponding sets to test them
    if options.test_set == 0 or options.test_set == 3:
        options.representer_path = root_representer + 'train_'
        _, loss, _, acc = net.test(options, trainX, trainY)
        test_loss.append(loss)
        test_acc.append(acc)
        sets.append('Train')
    if options.test_set == 1 or options.test_set == 3:
        options.representer_path = root_representer + 'val_'
        _, loss, _, acc = net.test(options, valX, valY)
        test_loss.append(loss)
        test_acc.append(acc)
        sets.append('Val')
    if options.test_set == 2 or options.test_set == 3:
        options.representer_path = root_representer + 'test_'
        _, loss, _, acc = net.test(options, testX, testY)
        test_loss.append(loss)
        test_acc.append(acc)
        sets.append('Test')
    options.representer_path = root_representer
    statfile = options.train_dir + "/evaluation.txt"
    with open(statfile, "w") as f:
        for i in range(len(sets)):
            line = "Set: %s | Loss: %s | Accuracy: %s\n" % (
                sets[i], test_loss[i], test_acc[i])
            f.write(line)


def compare_labels(gt_labels, pred_labels):
    """Compares the labels

    Arguments:
        gt_labels {array} -- ground truth labels
        pred_labels {array} -- predicted labels
    """
    for i in range(len(gt_labels)):
        print('Correct: %s | Predicted: %s' %
              (gt_labels[i], np.argmax(pred_labels[i])))

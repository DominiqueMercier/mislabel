import tensorflow as tf
from keras.layers import Conv1D, Dense, Dropout, Flatten, MaxPooling1D

def inference(input_placeholder, num_classes):
    """Generates a model with a given architecture

    Arguments:
        input_placeholder {tfplaceholder} -- Inpute shape
        num_classes {int} -- number of classes

    Returns:
        layers -- inference model layer
    """    
    layers = []

    with tf.variable_scope('conv1'):
        h_conv1 = Conv1D(32, 3, activation='relu', padding='same',
                         data_format='channels_first')(input_placeholder)
        layers.append(h_conv1)
    
    with tf.variable_scope('conv2'):
        h_conv2 = Conv1D(32, 3, activation='relu', padding='same',
                         data_format='channels_first')(h_conv1)
        layers.append(h_conv2)

    with tf.variable_scope('pool1'):
        h_pool1 = MaxPooling1D(2, strides=2, padding='same',
                               data_format='channels_first')(h_conv2)
        layers.append(h_pool1)

    with tf.variable_scope('conv3'):
        h_conv3 = Conv1D(64, 3, activation='relu', padding='same',
                         data_format='channels_first')(h_pool1)
        layers.append(h_conv3)

    with tf.variable_scope('conv4'):
        h_conv4 = Conv1D(64, 3, activation='relu', padding='same',
                         data_format='channels_first')(h_conv3)
        layers.append(h_conv4)

    with tf.variable_scope('pool2'):
        h_pool2 = MaxPooling1D(2, strides=2, padding='same',
                               data_format='channels_first')(h_conv4)
        layers.append(h_pool2)

    with tf.variable_scope('conv5'):
        h_conv5 = Conv1D(128, 3, activation='relu', padding='same',
                         data_format='channels_first')(h_pool2)
        layers.append(h_conv5)

    with tf.variable_scope('conv6'):
        h_conv6 = Conv1D(128, 3, activation='relu', padding='same',
                         data_format='channels_first')(h_conv5)
        layers.append(h_conv6)

    with tf.variable_scope('pool3'):
        h_pool3 = MaxPooling1D(2, strides=2, padding='same',
                               data_format='channels_first')(h_conv6)
        layers.append(h_pool3)

    with tf.variable_scope('flatten'):
        h_flat = Flatten(data_format='channels_first')(h_pool3)
        layers.append(h_flat)

    with tf.variable_scope('dense1'):
        h_dense1 = Dense(256, activation='relu')(h_flat)
        layers.append(h_dense1)

    with tf.variable_scope('dropout1'):
        h_drop1 = Dropout(0.5)(h_dense1)
        layers.append(h_drop1)

    with tf.variable_scope('dense2'):
        h_dense2 = Dense(256, activation='relu')(h_drop1)
        layers.append(h_dense2)

    with tf.variable_scope('dropout2'):
        h_drop2 = Dropout(0.5)(h_dense2)
        layers.append(h_drop2)

    with tf.variable_scope('dense3'):
        h_dense3 = Dense(num_classes)(h_drop2)
        layers.append(h_dense3)

    return layers

import numpy as np
import tensorflow as tf


def prepare_data(data_set):
    # change the target layer into one hot vectors
    data_set = data_set.map(lambda img, target: (img, tf.one_hot(target,10)))
    # change the datatype to float 32
    data_set = data_set.map(lambda img, target: (tf.cast(img, tf.float32), (tf.cast(target, tf.float32))))
    # normalize the image values (0:255), to (-1:1)
    data_set = data_set.map(lambda img, target: ((img / 128.) - 1., target))
    # shuffle the datasets
    data_set = data_set.shuffle(buffer_size=1000)
    # batch the datasets
    data_set = data_set.batch(16)
    data_set = data_set.prefetch(5)
    return data_set

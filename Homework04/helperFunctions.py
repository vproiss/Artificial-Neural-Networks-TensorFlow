"""
Modul for all helper functions

Created: 19.11.21, 13:06

Author: LDankert
"""
import tensorflow as tf


# Preprocessing function
def prepare_data(data_set):
    # convert data from uint8 to float32
    data_set = data_set.map(lambda wine, target: (tf.cast(wine, tf.float32), target))
    data_set = data_set.shuffle(buffer_size=1000)
    data_set = data_set.batch(10)
    return data_set

import tensorflow as tf
import numpy as np


# @tf.function
def preprocessing_data(dataset, batch_size):
    """ Preprocess the given dataset with a defined noise factor for the noise adding
    ----------
    dataset: respective dataset (tf.dataset)
        the dataset to preprocess
    Returns
    -------
    dataset: tf.dataset
        the preprocessed dataset
    """
    # change dtype into float 32
    dataset = dataset.map(lambda img, target: (tf.cast(img, tf.float32), tf.cast(target, tf.float32)))
    # reshape the images
    dataset = dataset.map(lambda img, target: (tf.reshape(img, [28, 28, 1]), target))
    # normalize the data
    dataset = dataset.map(lambda img, target: ((img / 128.) - 1, target))
    # shuffle the datasets
    dataset = dataset.shuffle(buffer_size=1000)
    # batch the datasets
    dataset = dataset.batch(batch_size)
    # prefetch the datasets
    dataset = dataset.prefetch(20)
    return dataset

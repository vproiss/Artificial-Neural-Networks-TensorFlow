"""
Preprocessing methods

Created: 11.12.21, 18:54

Author: LDankert
"""
import tensorflow as tf
import numpy as np


# Generator for white noise, needs two integer for sequence length and number of samples
def integration_task(seq_len, num_samples):
    for i in range(num_samples):
        # generating noise error
        noise = np.random.normal(size=seq_len)
        # calculates target
        target = np.int(np.sum(noise, axis=-1) >= 0)
        # handle empty dimensions
        noise = tf.expand_dims(noise, axis=-1)
        target = np.expand_dims(target, axis=-1)
        yield noise, target


# Function fo preprocessing the dataset
def preprocess_dataset(dataset):
    # shuffle the datasets
    dataset = dataset.shuffle(buffer_size=1000)
    # batch the datasets
    dataset = dataset.batch(10)
    # prefetch the datasets
    dataset = dataset.prefetch(12)
    return dataset


# @tf.function
def train_step(model, input, target, loss_function, optimizer):
    with tf.GradientTape() as tape:
        prediction = model(input)
        train_loss = loss_function(target, prediction[-1])
        sample_training_accuracy = np.round(target, 0) == np.round(prediction[-1][-1], 0)  # np.round(,0)
        sample_training_accuracy = np.mean(sample_training_accuracy)
        gradients = tape.gradient(train_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return train_loss, sample_training_accuracy


def test(model, test_data, loss_function):
    # test over complete test data
    test_accuracy_aggregator = []
    test_loss_aggregator = []

    for (input, target) in test_data:
        prediction = model(input)
        sample_test_loss = loss_function(target, prediction)  # tf.keras.losses.BinaryCrossentropy()
        sample_test_accuracy = np.round(target, 0) == np.round(prediction, 0)  # np.round(,0)
        sample_test_accuracy = np.mean(sample_test_accuracy)
        test_loss_aggregator.append(sample_test_loss.numpy())
        test_accuracy_aggregator.append(np.mean(sample_test_accuracy))

    test_loss = np.mean(test_loss_aggregator)
    test_accuracy = np.mean(test_accuracy_aggregator)

    return test_loss, test_accuracy

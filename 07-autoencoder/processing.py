import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# @tf.function
def preprocessing_data(dataset, noise_factor):
    """ Preprocess the given dataset with a defined noise factor for the noise adding
    ----------
    dataset: respective dataset (tf.dataset)
        the dataset to preprocess
    noise_factor: float
        the factor how strong the noise add should be
    Returns
    -------
    dataset: tf.dataset
        the preprocessed dataset
    """
    # change dtype into float 32
    dataset = dataset.map(lambda img, target: (tf.cast(img, tf.float32), target))
    # normalize the data
    dataset = dataset.map(lambda img, target: ((img / 256.), target))
    # add 3rd color row
    #dataset = dataset.map(lambda img, target: (tf.expand_dims(img, axis=-1), target))
    # change target into the original image
    dataset = dataset.map(lambda img, target: (img, img))
    # creates a random vector and
    dataset = dataset.map(lambda img, target: (img + noise_factor *
                                               tf.random.normal([28, 28, 1], mean=0.0, stddev=1.0, dtype=tf.float32), target))
    # clip the values between 0 and 1
    dataset = dataset.map(lambda img, target: (tf.clip_by_value(img, clip_value_min=0, clip_value_max=1), target))
    # shuffle the datasets
    dataset = dataset.shuffle(buffer_size=1000)
    # batch the datasets
    dataset = dataset.batch(200)
    # prefetch the datasets
    dataset = dataset.prefetch(20)
    return dataset


@tf.function
def train_step(model, input, target, loss_function, optimizer):
    """ Computes a train step with the given data
  Parameters
  ----------
  model : respective model class (super: tf.keras.Model)
    the model to perform train step
  inputs : tf.Tensor
    the input for the model
  target : tf.Tensor
    the target label
  loss_function : tf.keras.losses
    given loss function of respective tensorflow classes
  optimizer : tf.keras.optimizers
    given optimizer of respective tensorflow classes
  Returns
  -------
  loss : tf.Tensor
      the current loss of the model
  """
    with tf.GradientTape() as tape:
        prediction = model(input, training=True)
        #print(f"Target - Loss {tf.reduce_mean(target-prediction)}")
        loss = loss_function(target, prediction)
        #print(f"Actual loss:{loss}")
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, prediction, input



def test(model, test_data, loss_function):
    """ Computes a test step with the given data
  Parameters
  ----------
  model : respective model class (super: tf.keras.Model)
    the model to perform train step
  test_data : tf.Tensor
    the test dataset
  loss_function : tf.keras.losses
    given loss function of respective tensorflow classes
  Returns
  -------
  test_loss : tf.Tensor
      the current loss of the model
  test_accuracy : double
      the current loss of the model
  """
    test_accuracy_aggregator = np.empty(0)
    test_loss_aggregator = np.empty(0)

    for (input, target) in test_data:
        prediction = model(input, training=False)
        sample_test_loss = loss_function(target, prediction)
        sample_test_accuracy = np.argmax(target, axis=1) == np.argmax(prediction, axis=1)
        #sample_test_accuracy = prediction - target
        sample_test_accuracy = np.mean(sample_test_accuracy)
        #print(sample_test_accuracy)
        test_loss_aggregator = np.append(test_loss_aggregator, sample_test_loss)
        test_accuracy_aggregator = np.append(test_accuracy_aggregator, sample_test_accuracy)

    test_loss = tf.reduce_mean(test_loss_aggregator)
    test_accuracy = tf.reduce_mean(test_accuracy_aggregator)

    return test_loss, test_accuracy

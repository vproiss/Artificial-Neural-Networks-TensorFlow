import numpy as np
import tensorflow as tf
from model import LSTM_Model
from preprocessing import preprocess_dataset, integration_task, train_step, test

sequence_length = 20  # will be imported from preprocessing module
number_of_samples = 80000  # will be imported from preprocessing module


# Iterates over integration_task with specific values
def my_integration_task():
    yield next(integration_task(sequence_length, number_of_samples))


# generating datasets output_signature adds metadata to the dataset like dtype
ds = tf.data.Dataset.from_generator(my_integration_task, output_signature=(
    tf.TensorSpec(shape=(sequence_length, 1), dtype=tf.float32),
    tf.TensorSpec(shape=(1), dtype=tf.float32)))

ds = preprocess_dataset(ds)
train_dataset = ds.take(np.round(number_of_samples * 0.8))
test_dataset = ds.take(np.round(number_of_samples * 0.8))

tf.keras.backend.clear_session()

num_epochs = 50
learning_rate = 0.1
cross_entropy_loss = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate)

model = LSTM_Model()

train_losses = []
train_accuracies = []

test_losses = []
test_accuracies = []

# Testing once before we begin.
test_loss, test_accuracy = test(model, test_dataset, cross_entropy_loss)
test_losses.append(test_loss)
test_accuracies.append(test_accuracy)

# Check how model performs on train data once before we begin.
train_loss, train_accuracy = test(model, train_dataset, cross_entropy_loss)
train_losses.append(train_loss)
train_accuracies.append(train_accuracy)

for epoch in range(num_epochs):
    print(f'Epoch: {str(epoch)} starting with accuracy {test_accuracies[-1]}')

    # Training (and checking in with training).
    epoch_loss_agg = []
    epoch_acc_agg = []
    for input, target in train_dataset:
        train_loss, train_accuracy = train_step(model, input, target, cross_entropy_loss, optimizer)
        epoch_loss_agg.append(train_loss)
        epoch_acc_agg.append(train_accuracy)
    # optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    # Track training loss and accuracy.
    train_losses.append(tf.reduce_mean(epoch_loss_agg))
    train_accuracies.append(tf.reduce_mean(epoch_acc_agg))

    # Testing
    if epoch % 5 == 0:  # test only twice
        test_loss, test_accuracy = test(model, test_dataset, cross_entropy_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
    else:
        test_losses.append(0)  # to make the plot kinda right
        test_accuracies.append(0)

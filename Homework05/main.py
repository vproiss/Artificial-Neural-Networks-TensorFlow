import tensorflow as tf
import tensorflow_datasets as tfds
import os.path
from preprocessing import prepare_data


'''
# Import csv as data frame with pandas from link
if os.path.exists("fashion_mnist_train") and os.path.exists("fashion_mnist_test"):
    # Load data frame when local saved
    print("Loading data frame from local")
    ds_train = tf.data.experimental.load("fashion_mnist_train")
    ds_test = tf.data.experimental.load("fashion_mnist_test")
else:
    # Downloads data frame and safe it local
    print("Loading data frame from tfds")
    ds_train, ds_test = tfds.load('fashion_mnist', split = ['train','test'], as_supervised = True)
    tf.data.experimental.save(ds_train, "fashion_mnist_train")
    tf.data.experimental.save(ds_train, "fashion_mnist_test")
    '''
ds_train, info = tfds.load('fashion_mnist', split='train', as_supervised = True, with_info=True)

print(info)
#print(ds_test)

ds_train = ds_train.apply(prepare_data)
#ds_test = ds_test.apply(prepare_data)


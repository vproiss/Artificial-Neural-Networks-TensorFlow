import numpy as np
import pandas as pd
import os.path
import tensorflow as tf
from helperFunctions import prepare_data


# Import csv as data frame with pandas from link
if os.path.exists("wine_dataframe.csv"):
    # Load data frame when local saved
    print("Loading data frame from local")
    df = pd.read_csv("wine_dataframe.csv", index_col=0)
else:
    # Downloads data frame and safe it local
    print("Loading data frame from URL")
    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=";")
    df.to_csv("wine_dataframe.csv")

# separates the target column
target = df.pop('quality')
# convert it into a binary label
target[target < target.quantile(q=0.75)] = 0
target[target >= target.quantile(q=0.75)] = 1
# change it to float and convert it into a tensor
target = tf.cast(target, tf.float32)

# list of the feature names, we can just use all possible columns
feature_names = (df.columns.values)
# extracted feature values
feature_values = df[feature_names]

# creating the tensorflow dataset
full_ds = tf.data.Dataset.from_tensor_slices((feature_values, target))


# split the dataset into training, validation and test set
train_size = int(0.7 * len(df.index))
valid_size = int(0.15 * len(df.index))

train_ds = full_ds.take(train_size)
remaining = full_ds.skip(train_size)
valid_ds = remaining.take(valid_size)
test_ds = remaining.skip(valid_size)


# Apply the preprocessing to all datasets
train_ds = train_ds.apply(prepare_data)
valid_ds = valid_size.apply(prepare_data)
test_ds = test_ds.apply(prepare_data)


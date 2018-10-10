# Liner regression model for predicting housing prices in California
# based on code from the Google Machine Learning Crash Course
# by: Stephan N. Ofosuhene


import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset
# import tensorflow.include.

print(tf.__version__)

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

#%%

housing_data = pd.read_csv('/Users/stephanofosuhene/Documents/Documents - Stephan’s MacBook Pro/Code/tf_python/'
                           'practice/data/california_housing_data.csv')

#%%


# normalize the data
housing_data['median_house_value'] /= 1000

print(housing_data.head())
# print(housing_data.describe())

#%%

# randomize the data to eliminate ordering effects that may affect stochastic gradient descent
housing_data = housing_data.reindex(
    np.random.permutation(housing_data.index))

#%%
# Extract required features
num_rooms = housing_data[['total_rooms']]
# print(num_rooms)

# configure feature columns
feature_column = [tf.feature_column.numeric_column("total_rooms")]


# extract targets
targets = housing_data['median_house_value']

# setup gradient descent optimizer
grad_descent_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0000001)
optimizer = tf.contrib.estimator.clip_gradients_by_norm(grad_descent_optimizer, 5.0)


# configure the gradient descent model
linear_regressor = tf.estimator.LinearRegressor(
    feature_columns=feature_column,
    optimizer=optimizer
)



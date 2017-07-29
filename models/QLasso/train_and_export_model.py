import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.learn import LinearRegressor
from tensorflow.contrib import layers
from tensorflow.contrib.learn.python.learn.utils import input_fn_utils
tf.logging.set_verbosity(tf.logging.INFO)

import pandas as pd


def model(features, targets, mode):
    W = tf.get_variable("W", [1, len(features)], dtype=tf.float64)
    b = tf.get_variable("b", [1], dtype=tf.float64)
    X = [features[key] for key in features]
    y = tf.reshape(tf.matmul(W, X) + b, [-1])

    loss = tf.reduce_mean(tf.square(y - targets)) + tf.reduce_sum(tf.abs(W))
    global_step = tf.train.get_global_step()
    optimizer = tf.train.GradientDescentOptimizer(0.001)
    train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))

    return tf.contrib.learn.ModelFnOps(
        mode=mode, predictions=y,
        loss=loss,
        train_op=train)

FEATURES = ["ttt30", "all", "MEP", "triaged", "PRIO3", "PRIO4"]

pdframe = pd.read_csv("QLasso2017-07-28T11:36:47.045+02:00.csv")

def input_fn_train():
    feature_cols = {name: tf.constant(pdframe[name].get_values() / max(pdframe[name].get_values()), dtype=tf.float64) for name in FEATURES}
    outputs = tf.constant(pdframe["TTLOfNextPatient"].get_values(), dtype=tf.float64)
    #outputs = outputs / tf.reduce_max(outputs)
    return feature_cols, outputs

feature_cols = [layers.real_valued_column(name) for name in FEATURES]
'''
regressor = LinearRegressor(
    feature_columns=feature_cols,
    model_dir="./modeldir"
)
'''
regressor = learn.Estimator(model_fn=model, model_dir="./modeldir")
regressor.fit(input_fn=input_fn_train, steps=10000)

'''
def serving_input_fn():
    default_inputs = {col.name: tf.placeholder(col.dtype, [None]) for col in feature_cols}
    features = {key: tf.expand_dims(tensor, -1) for key, tensor in default_inputs.items()}
    return input_fn_utils.InputFnOps(
        features=features,
        labels=None,
        default_inputs=default_inputs
    )

regressor.export_savedmodel(
   "exportedmodel",
    serving_input_fn
)
'''

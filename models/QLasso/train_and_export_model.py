# -*- coding: utf-8 -*-

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
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))

    return tf.contrib.learn.ModelFnOps(
        mode=mode, predictions=y,
        loss=loss,
        train_op=train)

#FEATURES = ["ttt30", "all", "MEP", "triaged", "PRIO3", "PRIO4"]
WAIT_TIME_FEATURES = ["ttt30", "ttl30", "ttk30", "ttt60", "ttl60", "ttk60", "ttt120", "ttl120", "ttk120"]
# to be picky one should calculate e.g. untriaged = all - triaged, but regression weights can be negative so leaving it for now
WORKLOAD_FEATURES = ["UntreatedLowPrio", "all", "MEP", "triaged", "metdoctor", "done", "PRIO1", "PRIO2", "PRIO3", "PRIO4", "PRIO5"]
CAPACITY_FEATURES = ["doctors60", "teams60"]

FEATURES = WAIT_TIME_FEATURES + WORKLOAD_FEATURES + CAPACITY_FEATURES

pdframe = pd.read_csv("QLasso2017-07-28T11:36:47.045+02:00.csv")

def generate_Q_features(frame, workload_features, capacity_features):
    Q_features = {}
    for workload in workload_features:
        for capacity in capacity_features:
            if capacity == 0: # pretty arbitrary constant 0.5 to avoid division by 0
                Q_features[workload + "/" + capacity] = frame[workload].get_values() / 0.5
            else:
                Q_features[workload + "/" + capacity] = frame[workload].get_values() / frame[capacity].get_values()
    return Q_features

def input_fn_train():
    #feature_cols = {name: tf.constant(pdframe[name].get_values() / max(pdframe[name].get_values()), dtype=tf.float64) for name in FEATURES}
    feature_cols = {}
    Q_features = generate_Q_features(pdframe, WORKLOAD_FEATURES, CAPACITY_FEATURES)
    for key in Q_features:
        col = tf.constant(Q_features[key], dtype=tf.float64)
        feature_cols[key] = col / tf.reduce_max(col) # normalization
    for feature in WAIT_TIME_FEATURES:
        col = tf.constant(pdframe[feature].get_values(), dtype=tf.float64)
        feature_cols[feature] = col / tf.reduce_max(col) # normalization
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

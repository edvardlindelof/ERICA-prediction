# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.learn import LinearRegressor
from tensorflow.contrib import layers
from tensorflow.contrib.learn.python.learn.utils import input_fn_utils
tf.logging.set_verbosity(tf.logging.INFO)

import pandas as pd
import numpy as np

from time_of_week_feature import to_time_of_week_feature, to_time_of_week_feature_np


# TODO plain model stuff into own file

def model(features, targets, mode):
    W = tf.get_variable("W", [1, len(features)])
    b = tf.get_variable("b", [1])
    X = [features[key] for key in features]
    y = tf.reshape(tf.matmul(W, X) + b, [-1])

    #un_penaltied_loss = tf.reduce_mean(tf.square(y - targets))
    un_penaltied_loss = tf.losses.mean_squared_error(targets, y)
    #loss = un_penaltied_loss + 100.0 * tf.norm(W, ord=1) # TODO penalty hyperparameter
    loss = un_penaltied_loss
    global_step = tf.train.get_global_step()
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))

    mse_minutes = tf.div(un_penaltied_loss, 3600, name="mse_minutes")
    tf.summary.scalar("mse_in_minutes", mse_minutes)
    zero = tf.constant(0, dtype=tf.float32)
    #non_zero_weights = tf.not_equal(W, zero)
    non_zero_weights = tf.greater(tf.abs(W), zero + 0.3)
    n_non_zero_weights = tf.reduce_sum(tf.cast(non_zero_weights, tf.float32), name="n_non_zero_weights")
    tf.summary.scalar("non-zero_weights", n_non_zero_weights)
    #for i in range(len(features)):
    #    tf.summary.scalar("W_element" + str(i), W[0, i])

    return tf.contrib.learn.ModelFnOps(
        mode=mode,
        predictions=y,
        loss=loss,
        train_op=train
    )

WAIT_TIME_FEATURES = ["ttt30", "ttl30", "ttk30", "ttt60", "ttl60", "ttk60", "ttt120", "ttl120", "ttk120"]
# to be picky one should calculate e.g. untriaged = all - triaged, but regression weights can be negative so leaving it for now
WORKLOAD_FEATURES = ["UntreatedLowPrio", "all", "MEP", "triaged", "metdoctor", "done", "PRIO1", "PRIO2", "PRIO3", "PRIO4", "PRIO5"]
CAPACITY_FEATURES = ["doctors60", "teams60"]

FEATURES = WAIT_TIME_FEATURES + WORKLOAD_FEATURES + CAPACITY_FEATURES

pdframe = pd.read_csv("QLasso2017-08-04T16:02:58.809+02:00.csv")

# doing this outside of input_fn_train bc if placed in there it will be called maaaaaaany times
epoch_seconds = np.array(pdframe["epochseconds"].get_values(), dtype=np.int32)
time_of_week_feature = to_time_of_week_feature_np(epoch_seconds)
time_of_week_feature = time_of_week_feature / np.max(time_of_week_feature)

def generate_Q_features(frame, workload_features, capacity_features):
    Q_features = {}
    for workload in workload_features:
        for capacity in capacity_features:
            load = tf.constant(frame[workload].get_values(), dtype=tf.float32)
            cap = tf.constant(frame[capacity].get_values(), dtype=tf.float32)
            where_cap_small = tf.cast(tf.less(cap, 0.99), tf.float32)
            min_bound_cap = cap + 0.5 * where_cap_small * tf.ones_like(cap) # replace 0s with 0.5s
            Q_features[workload + "/" + capacity] = load / min_bound_cap
    return Q_features

def input_fn_train():
    feature_cols = {}

    ttl_next_low_prio_patient = tf.constant(pdframe["TTLOfNextPatient"].get_values(), dtype=tf.float32)

    epoch_seconds = tf.constant(pdframe["epochseconds"].get_values(), dtype=tf.int32)
    # slows down step time with a factor of about 5 despite only needing to be called once..
    #time_of_week_feature = to_time_of_week_feature(epoch_seconds)
    #feature_cols["TimeOfWeekFeature"] = time_of_week_feature / tf.reduce_max(time_of_week_feature)
    feature_cols["TimeOfWeekFeature"] = tf.constant(time_of_week_feature)

    untreated_low_prio_col = tf.constant(pdframe["UntreatedLowPrio"].get_values(), dtype=tf.float32)
    feature_cols["UntreatedLowPrio"] = untreated_low_prio_col / tf.reduce_max(untreated_low_prio_col) # normalization

    for workload in WORKLOAD_FEATURES:
        load = tf.constant(pdframe[workload].get_values(), dtype=tf.float32)
        feature_cols[workload] = load / tf.reduce_max(load)

    for feature in WAIT_TIME_FEATURES:
        col = tf.constant(pdframe[feature].get_values(), dtype=tf.float32)
        feature_cols[feature] = col / tf.reduce_max(col) # normalization

    Q_features = generate_Q_features(pdframe, WORKLOAD_FEATURES, CAPACITY_FEATURES)
    for key in Q_features:
        col = Q_features[key]
        feature_cols[key] = col / tf.reduce_max(col) # normalization

    outputs = ttl_next_low_prio_patient
    #outputs = outputs / tf.reduce_max(outputs)
    return feature_cols, outputs

regressor = learn.Estimator(model_fn=model, model_dir="./modeldir")
print_tensor = learn.monitors.PrintTensor(["n_non_zero_weights", "mse_minutes", "W"])
regressor.fit(input_fn=input_fn_train, steps=50000, monitors=[print_tensor])

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

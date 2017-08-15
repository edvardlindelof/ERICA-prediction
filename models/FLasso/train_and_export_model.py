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


def model(features, targets, mode):
    target_keys = targets.keys()

    W = tf.get_variable("W", [len(targets), len(features)])
    b = tf.get_variable("b", [len(targets), 1])
    X = [features[key] for key in features]
    print X
    print tf.matmul(W, X)
    print b
    y = tf.matmul(W, X) + b
    t = tf.reshape([targets[key] for key in target_keys], [len(targets), -1])
    print y[:, 1]

    #un_penaltied_loss = tf.reduce_mean(tf.square(y - targets))
    un_penaltied_loss = tf.losses.mean_squared_error(t, y)
    # TODO penalty hyperparameter
    # TODO perhaps make with one penalty for each output
    loss = un_penaltied_loss + (0.1 / len(targets)) * tf.norm(W, ord=1)
    #loss = un_penaltied_loss
    global_step = tf.train.get_global_step()
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))

    mse = tf.identity(un_penaltied_loss, name="mse")
    tf.summary.scalar("mean_square_error", mse)

    for i in range(len(target_keys)):
        zero = tf.constant(0, dtype=tf.float32)
        #non_zero_weights = tf.not_equal(W, zero)
        non_zero_weights = tf.greater(tf.abs(W[i]), zero + 0.3)
        n_non_zero_weights = tf.reduce_sum(tf.cast(non_zero_weights, tf.float32), name=target_keys[i] + "-n_non_zero_weights")
        title = target_keys[i]
        tf.summary.scalar(fix(title) + "-non-zero_weights", n_non_zero_weights)

    return tf.contrib.learn.ModelFnOps(
        mode=mode,
        #predictions=y,
        predictions={target_keys[i]: y[i] for i in range(len(target_keys))},
        loss=loss,
        train_op=train
    )

def fix(title):
    if title == "Kölapp":
        return "Kolapp"
    if title == "Läkare":
        return "Lakare"
    else:
        return title

FREQUENCY_FEATURES = [
    "Kölapp30","Triage30","Läkare30","Klar30",
    "Kölapp60","Triage60","Läkare60","Klar60",
    "Kölapp120","Triage120","Läkare120","Klar120"
]
# to be picky one should calculate e.g. untriaged = all - triaged, but regression weights can be negative so leaving it for now
WORKLOAD_FEATURES = ["UntreatedLowPrio", "all", "MEP", "triaged", "metdoctor", "done", "PRIO1", "PRIO2", "PRIO3", "PRIO4", "PRIO5"]
CAPACITY_FEATURES = ["doctors60", "teams60"]

FEATURES = FREQUENCY_FEATURES + WORKLOAD_FEATURES + CAPACITY_FEATURES

pdframe = pd.read_csv("FLasso2017-08-14T15:29:28.093+02:00.csv")
event_titles = ["Kölapp", "Triage"] # Kölapp, Triage, Klar or Läkare

# doing this outside of input_fn_train bc if placed in there it will be called maaaaaaany times
epoch_seconds = np.array(pdframe["epochseconds"].get_values(), dtype=np.int32)
time_of_week_features = {}
for event_title in event_titles:
    time_of_week_feature = to_time_of_week_feature_np(epoch_seconds, event_title)
    time_of_week_feature = time_of_week_feature / np.max(time_of_week_feature)
    time_of_week_features[fix(event_title)] = time_of_week_feature

def generate_Q_features(frame, workload_features, capacity_features):
    Q_features = {}
    for workload in workload_features:
        for capacity in capacity_features:
            load = tf.constant(frame[workload].get_values(), dtype=tf.float32)
            cap = tf.constant(frame[capacity].get_values(), dtype=tf.float32)
            where_load_small = tf.cast(tf.less(load, 0.99), tf.float32)
            min_bound_load = load + 0.5 * where_load_small * tf.ones_like(load) # replace 0s with 0.5s
            Q_features[capacity + "/" + workload] = cap / min_bound_load
    return Q_features

def input_fn_train():
    feature_cols = {}

    # epoch_seconds = tf.constant(pdframe["epochseconds"].get_values(), dtype=tf.int32)

    for key in time_of_week_features:
        feature_cols[key + "TimeOfWeekFeature"] = tf.constant(time_of_week_features[key])

    untreated_low_prio_col = tf.constant(pdframe["UntreatedLowPrio"].get_values(), dtype=tf.float32)
    feature_cols["UntreatedLowPrio"] = untreated_low_prio_col / tf.reduce_max(untreated_low_prio_col) # normalization

    for workload in WORKLOAD_FEATURES:
        load = tf.constant(pdframe[workload].get_values(), dtype=tf.float32)
        feature_cols[workload] = load / tf.reduce_max(load)

    for feature in FREQUENCY_FEATURES:
        col = tf.constant(pdframe[feature].get_values(), dtype=tf.float32)
        feature_cols[feature] = col / tf.reduce_max(col) # normalization

    Q_features = generate_Q_features(pdframe, WORKLOAD_FEATURES, CAPACITY_FEATURES)
    for key in Q_features:
        col = Q_features[key]
        feature_cols[key] = col / tf.reduce_max(col) # normalization

    outputs = {}
    for event_title in event_titles:
        next_hour_values = tf.constant(pdframe["NextHour" + event_title].get_values(), dtype=tf.float32)
        outputs["NextHour" + fix(event_title)] = next_hour_values

    #outputs = outputs / tf.reduce_max(outputs)
    return feature_cols, outputs

regressor = learn.Estimator(model_fn=model, model_dir="./modeldir")
print_tensor = learn.monitors.PrintTensor(["NextHourTriage-n_non_zero_weights", "mse", "W"])
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

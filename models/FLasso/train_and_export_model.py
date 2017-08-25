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
    #W = tf.placeholder(tf.float32)
    #b = tf.placeholder(tf.float32)
    W = tf.get_variable("W", [len(EVENT_TITLES), len(features)])
    b = tf.get_variable("b", [len(EVENT_TITLES), 1])
    #X = [features[key] for key in features]
    X = tf.reshape([features[key] for key in features], [len(features), -1])
    #t = tf.reshape([targets[key] for key in target_keys], [len(targets), -1])

    target_keys = []
    if mode == "train":
        target_keys = targets.keys()
        t = tf.reshape([targets[key] for key in target_keys], [len(targets), -1])
    else:
        t = tf.placeholder(tf.float32)

    print W
    print tf.reshape(X, [len(features), -1])
    y = tf.matmul(W, X) + b

    print "inside model()"
    print targets
    print mode
    print mode == "infer"
    print t
    print y
    print W
    print FEATURES
    print [feature for feature in features.keys()]

    un_penaltied_loss = tf.reduce_mean(tf.square(y - t), 1)
    #un_penaltied_loss = tf.losses.mean_squared_error(t, y)
    # TODO penalty hyperparameter
    # TODO perhaps make vector with one penalty for each output
    loss = tf.reduce_mean(un_penaltied_loss) + (0.1 / tf.cast(tf.size(t), tf.float32)) * tf.norm(W, ord=1)
    #loss = un_penaltied_loss
    global_step = tf.train.get_global_step()
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))

    for i in range(len(target_keys)):
        mse = tf.identity(un_penaltied_loss[i], name=target_keys[i] + "-mse")
        tf.summary.scalar(target_keys[i] + "-mean_square_error", mse)

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
        predictions=tf.reduce_sum(y),
        #predictions={target_keys[i]: y[i] for i in range(len(target_keys))},
        loss=loss,
        train_op=train,
        #output_alternatives=["NextHourLakare"]
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
#EVENT_TITLES = ["Triage"] # Kölapp, Triage, Klar or Läkare
EVENT_TITLES = ["Kölapp", "Triage", "Läkare", "Klar"]

# doing this outside of input_fn_train bc if placed in there it will be called maaaaaaany times
epoch_seconds = np.array(pdframe["epochseconds"].get_values(), dtype=np.int32)
time_of_week_features = {}
for event_title in EVENT_TITLES:
    time_of_week_feature = to_time_of_week_feature_np(epoch_seconds, event_title)
    time_of_week_feature = time_of_week_feature / np.max(time_of_week_feature)
    time_of_week_features[fix(event_title)] = time_of_week_feature

def generate_Q_features(workload_features, capacity_features):
    Q_features = {}
    for workload in workload_features:
        for capacity in capacity_features:
            load = workload_features[workload]
            cap = capacity_features[capacity]
            where_load_small = tf.cast(tf.less(load, 0.99), tf.float32)
            min_bound_load = load + 0.5 * where_load_small * tf.ones_like(load) # replace 0s with 0.5s
            Q_features[capacity + "/" + workload] = cap / min_bound_load
    return Q_features

def feature_engineering(workload_features, frequency_features, capacity_features, time_of_week_features):
    feature_cols = {}

    for key in time_of_week_features:
        feature_cols[key + "TimeOfWeekFeature"] = tf.constant(time_of_week_features[key])

    for key in workload_features:
        feature_cols[key] = workload_features[key]

    for key in frequency_features:
        feature_cols[key] = frequency_features[key]

    Q_features = generate_Q_features(workload_features, capacity_features)
    for key in Q_features:
        feature_cols[key] = Q_features[key]

    return feature_cols

def normalize_tensors(dict):
    normalization_constants = {key: tf.reduce_max(dict[key]) for key in dict}
    normalized_dict = {key: dict[key] / normalization_constants[key] for key in dict}
    return normalized_dict, normalization_constants

def input_fn_train():
    frequency_features = {}
    for feature in FREQUENCY_FEATURES:
        col = tf.constant(pdframe[feature].get_values(), dtype=tf.float32)
        frequency_features[feature] = col

    workload_features = {}
    for workload in WORKLOAD_FEATURES:
        load = tf.constant(pdframe[workload].get_values(), dtype=tf.float32)
        workload_features[workload] = load

    capacity_features = {}
    for key in CAPACITY_FEATURES:
        capacity_features[key] = tf.constant(pdframe[key].get_values(), dtype=tf.float32)

    feature_cols = feature_engineering(workload_features, frequency_features, capacity_features, time_of_week_features)
    normalized_feature_cols, normalization_constants = normalize_tensors(feature_cols)

    outputs = {}
    for event_title in EVENT_TITLES:
        next_hour_values = tf.constant(pdframe["NextHour" + event_title].get_values(), dtype=tf.float32)
        outputs["NextHour" + fix(event_title)] = next_hour_values

    return normalized_feature_cols, outputs

#regressor = learn.Estimator(model_fn=model, model_dir="./modeldir")
regressor = learn.Estimator(model_fn=model, model_dir="./modeldir")
w_to_monitor = ["NextHour" + fix(title) + "-n_non_zero_weights" for title in EVENT_TITLES]
mse_to_monitor = ["NextHour" + fix(title) + "-mse" for title in EVENT_TITLES]
#print_tensor = learn.monitors.PrintTensor(["NextHourTriage-n_non_zero_weights", "NextHourTriage-mse"])
print_tensor = learn.monitors.PrintTensor(w_to_monitor + mse_to_monitor)
#regressor.fit(input_fn=input_fn_train, steps=50000, monitors=[print_tensor])
regressor.fit(input_fn=input_fn_train, steps=450, monitors=[print_tensor])

def serving_input_fn():
    default_inputs = {fname: tf.placeholder(tf.float32) for fname in FEATURES}
    features = {key: tf.expand_dims(tensor, -1) for key, tensor in default_inputs.items()}
    return input_fn_utils.InputFnOps(
        features=features,
        labels=None,
        #labels=tf.constant([1]),
        default_inputs=default_inputs
    )

'''
regressor.export_savedmodel(
   export_dir_base="exportedmodel",
    serving_input_fn=serving_input_fn,
    #default_output_alternative_key="NextHourLakare"
    #default_output_alternative_key=["NextHourLakare"]
)
'''

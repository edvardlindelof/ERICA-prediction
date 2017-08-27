# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.learn import LinearRegressor
from tensorflow.contrib import layers
from tensorflow.contrib.learn.python.learn.utils import input_fn_utils
tf.logging.set_verbosity(tf.logging.INFO)

import pandas as pd
import numpy as np

from time_of_week_feature import to_time_of_week_feature, to_time_of_week_feature_np, _incidence_bins


def model(features_in, targets, mode):
    for key in features_in:
        if key != "epochseconds":
            features_in[key] = features_in[key] / feature_maxes[key] # normalization

    if mode == "train":
        time_of_week_features = TIME_OF_WEEK_FEATURES
    else:
        epoch_seconds = tf.cast(features_in["epochseconds"], tf.int32)
        time_of_week_features = {}
        for event_title in EVENT_TITLES:
            #time_of_week_feature = to_time_of_week_feature(epoch_seconds, event_title)
            #time_of_week_feature = time_of_week_feature / tf.reduce_max(time_of_week_feature)
            #time_of_week_features[fix(event_title)] = time_of_week_feature
            time_of_week_features[fix(event_title)] = tf.constant(1000., dtype=tf.float32)

    print "=================== " + "time_of_week_features:"
    print time_of_week_features

    features = feature_engineering(features_in, time_of_week_features)
    #print features
    #print len(features)
    #import sys
    #sys.exit()
    # TODO if normalization constants defined, use them instead
    #features, normalization_constants = normalize_tensors(unnormalized_features)

    #feature_keys = sorted(features.keys()) # I want to be sure of the order
    #normalization_constantsz = tf.stack([tf.reduce_max(features[key]) for key in feature_keys])
    #print normalization_constantsz
    #normalization_constants = tf.get_variable("normalization_constants", [len(features), 1])
    #normalization_constants.assign(tf.reshape(normalization_constantsz, [51, 1]), use_locking=True)
    #normalization_constants = tf.placeholder(tf.float32, [len(features)], "normalization_constants")
    #norms = tf.constant(normalization_constantsz, name="normalization_constants")

    '''
    i = 0
    for key in feature_keys:
        print tf.reduce_max(features[key])
        features[key] = features[key] / tf.reduce_max(features[key])
        #print norms[i]
        #features[key] = features[key] / norms[i]
        i += 1
    '''

    #W = tf.placeholder(tf.float32)
    #b = tf.placeholder(tf.float32)
    W = tf.get_variable("W", [len(EVENT_TITLES), len(features)])
    b = tf.get_variable("b", [len(EVENT_TITLES), 1])
    X = [features[key] for key in features]
    X = tf.reshape(X, [len(features), -1])
    #X = tf.div(tf.reshape([features[key] for key in feature_keys], [len(features), -1]), normalization_constants)
    #X = tf.reshape(tf.div([features[key] for key in feature_keys], normalization_constants), [len(features), -1])
    '''
    X = [features[key] for key in features]
    X = tf.cast(X, tf.float64)
    normalization_constants = tf.cast(normalization_constants, tf.float64)
    X = tf.div(tf.reshape(X, [len(features), -1]), normalization_constants)
    X = tf.cast(X, tf.float32)
    '''
    #t = tf.reshape([targets[key] for key in target_keys], [len(targets), -1])

    target_keys = []
    if mode == "train":
        target_keys = targets.keys()
        t = tf.reshape([targets[key] for key in target_keys], [len(targets), -1])
    else:
        t = tf.placeholder(tf.float32)

    print "inside model()"
    print tf.reshape(X, [len(features), -1])
    y = tf.matmul(W, X) + b

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

    print tf.size(X[0])
    n_inputs = tf.size(X[0])
    print tf.reshape(y, [n_inputs, -1])
    transp_y = tf.transpose(y) # bc ml-engine says outer dimension must be unknown
    #print transp_y[0]
    #import sys
    #sys.exit()
    reshp_y = tf.reshape(y, [n_inputs, -1])

    return tf.contrib.learn.ModelFnOps(
        mode=mode,
        #predictions=y[0],
        predictions=transp_y,
        #predictions=reshp_y,
        #predictions=tf.constant([1., 1.]),
        #predictions=tf.reduce_sum(y), # TODO outputting the dict below would be neater if it can be made to work
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

EPOCHSECONDS_FEATURE = ["epochseconds"]
FREQUENCY_FEATURES = [
    "Kölapp30","Triage30","Läkare30","Klar30",
    "Kölapp60","Triage60","Läkare60","Klar60",
    "Kölapp120","Triage120","Läkare120","Klar120"
]
# to be picky one should calculate e.g. untriaged = all - triaged, but regression weights can be negative so leaving it for now
WORKLOAD_FEATURES = ["UntreatedLowPrio", "all", "MEP", "triaged", "metdoctor", "done", "PRIO1", "PRIO2", "PRIO3", "PRIO4", "PRIO5"]
CAPACITY_FEATURES = ["doctors60", "teams60"]

FEATURES = EPOCHSECONDS_FEATURE + FREQUENCY_FEATURES + WORKLOAD_FEATURES + CAPACITY_FEATURES

pdframe = pd.read_csv("FLasso2017-08-14T15:29:28.093+02:00.csv")
#EVENT_TITLES = ["Triage"] # Kölapp, Triage, Klar or Läkare
EVENT_TITLES = ["Kölapp", "Triage", "Läkare", "Klar"]

# doing this outside of input_fn_train bc if placed in there it will be called maaaaaaany times
epoch_seconds = np.array(pdframe["epochseconds"].get_values(), dtype=np.int32)
TIME_OF_WEEK_FEATURES = {}
for event_title in EVENT_TITLES:
    time_of_week_feature = to_time_of_week_feature_np(epoch_seconds, event_title)
    time_of_week_feature = time_of_week_feature / np.max(time_of_week_feature)
    TIME_OF_WEEK_FEATURES[fix(event_title)] = time_of_week_feature

'''
WEEK_IN_SECONDS = 3600 * 24 * 7
BIN_WIDTH = 600 # in seconds
N_BINS = WEEK_IN_SECONDS / BIN_WIDTH

incidence_bins = {i: [] for i in range(N_BINS)}
epochsecond = np.array(pdframe["epochseconds"].get_values(), dtype=np.int32)
#next_hour_incidence = np.array(pdframe["NextHour" + event_title].get_values(), dtype=np.float32)
next_hour_incidence = np.array(pdframe["NextHour" + "Triage"].get_values(), dtype=np.float32)
for i in range(len(epochsecond)):
    second_in_week = epochsecond[i] % WEEK_IN_SECONDS
    bin_number = second_in_week / BIN_WIDTH
    incidence_bins[bin_number] = incidence_bins[bin_number] + [next_hour_incidence[i]]
print incidence_bins
incidence_mean_bins = np.array([np.mean(incidence_bins[i]) for i in range(N_BINS)], dtype=np.float32)
'''


feature_maxes = {}
for feature in FEATURES:
    values = np.array(pdframe[feature].get_values(), dtype=np.float32)
    feature_maxes[feature] = np.max(values)


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

def feature_engineering(features, time_of_week_features):
    feature_cols = {}

    #for key in time_of_week_features: # TODO time of week feature currently turned off!!
        #feature_cols[key + "TimeOfWeekFeature"] = time_of_week_features[key]

    workload_features = {}
    frequency_features = {}
    capacity_features = {}
    for key in features:
        if key in WORKLOAD_FEATURES:
            workload_features[key] = features[key]
        elif key in FREQUENCY_FEATURES:
            frequency_features[key] = features[key]
        elif key in CAPACITY_FEATURES:
            capacity_features[key] = features[key]

        if key != "epochseconds":
            feature_cols[key] = features[key]

    Q_features = generate_Q_features(workload_features, capacity_features)
    for key in Q_features:
        feature_cols[key] = Q_features[key]

    return feature_cols

def normalize_tensors(dict):
    normalization_constants = {key: tf.reduce_max(dict[key]) for key in dict}
    normalized_dict = {key: dict[key] / normalization_constants[key] for key in dict}
    return normalized_dict, normalization_constants

def input_fn_train():
    features = {}
    for key in WORKLOAD_FEATURES + FREQUENCY_FEATURES + CAPACITY_FEATURES:
        features[key] = tf.constant(pdframe[key].get_values(), dtype=tf.float32)

    #feature_cols = feature_engineering(features, time_of_week_features)
    #normalized_feature_cols, normalization_constants = normalize_tensors(feature_cols)

    outputs = {}
    for event_title in EVENT_TITLES:
        next_hour_values = tf.constant(pdframe["NextHour" + event_title].get_values(), dtype=tf.float32)
        outputs["NextHour" + fix(event_title)] = next_hour_values

    #return normalized_feature_cols, outputs
    return features, outputs

'''
training_feature_cols, _ = input_fn_train()
training_feature_keys = sorted(training_feature_cols.keys()) # I want to be sure of the order

maxes = [tf.reduce_max(training_feature_cols[key]) for key in training_feature_keys]
normalization_constants = tf.constant(tf.reshape(maxes, [len(maxes)]))
'''

#regressor = learn.Estimator(model_fn=model, model_dir="./modeldir")
regressor = learn.Estimator(model_fn=model, model_dir="./modeldir")
w_to_monitor = ["NextHour" + fix(title) + "-n_non_zero_weights" for title in EVENT_TITLES]
mse_to_monitor = ["NextHour" + fix(title) + "-mse" for title in EVENT_TITLES]
#print_tensor = learn.monitors.PrintTensor(["NextHourTriage-n_non_zero_weights", "NextHourTriage-mse"])
print_tensor = learn.monitors.PrintTensor(w_to_monitor + mse_to_monitor)
#regressor.fit(input_fn=input_fn_train, steps=50000, monitors=[print_tensor])
regressor.fit(input_fn=input_fn_train, steps=150, monitors=[print_tensor])

def input_fn_test():
    datapoint = {
        "epochseconds": 1503668054,
        "Kölapp30": 1, "Triage30": 1, "Läkare30": 1, "Klar30": 1,
        "Kölapp60": 1, "Triage60": 1, "Läkare60": 1, "Klar60": 1,
        "Kölapp120": 1, "Triage120": 1, "Läkare120": 1, "Klar120": 1,
        "UntreatedLowPrio": 1, "all": 1, "MEP": 1, "triaged": 1, "metdoctor": 1, "done": 1,
        "PRIO1": 1, "PRIO2": 1, "PRIO3": 1, "PRIO4": 1, "PRIO5": 1,
        "doctors60": 1, "teams60": 1
    }
    for key in datapoint:
        if key == "epochseconds":
            datapoint[key] = tf.constant(datapoint[key], dtype=tf.int32)
        else:
            datapoint[key] = tf.constant(datapoint[key], dtype=tf.float32)
    return datapoint

def serving_input_fn():
    #default_inputs = {fname: tf.placeholder(tf.float32) for fname in FEATURES}
    #features = {key: tf.expand_dims(tensor, -1) for key, tensor in default_inputs.items()}
    #default_inputs = {fname: tf.placeholder(tf.float32, [None]) for fname in FEATURES}
    #features = {key: tf.expand_dims(tensor, -1) for key, tensor in default_inputs.items()}
    default_inputs = {fname: tf.placeholder(tf.float32) for fname in FEATURES}
    features = {key: tf.reshape(tensor, [-1]) for key, tensor in default_inputs.items()}
    return input_fn_utils.InputFnOps(
        features=features,
        #labels=tf.placeholder(tf.float32,[4]),
        labels=None,
        #labels=tf.constant([1]),
        default_inputs=default_inputs
    )

regressor.export_savedmodel(
   export_dir_base="exportedmodel",
    serving_input_fn=serving_input_fn,
    #default_output_alternative_key="NextHourLakare"
    #default_output_alternative_key=["NextHourLakare"]
)

pred = regressor.predict(input_fn=input_fn_test)
print [p for p in pred]

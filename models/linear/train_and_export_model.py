import tensorflow as tf
from tensorflow.contrib.learn import LinearRegressor
from tensorflow.contrib import layers
from tensorflow.contrib.learn.python.learn.utils import input_fn_utils
tf.logging.set_verbosity(tf.logging.INFO)

import pandas as pd

FEATURES = ["ttt30", "all", "MEP", "triaged", "PRIO3", "PRIO4"]

pdframe = pd.read_csv("../NALState2017-07-26T18:45:33.190+02:00.csv")

def input_fn_train():
    feature_cols = {name: tf.constant(pdframe[name].get_values()) for name in FEATURES}
    outputs = tf.constant(pdframe["TTLOfNextPatient"].get_values())
    return feature_cols, outputs

feature_cols = [layers.real_valued_column(name) for name in FEATURES]
regressor = LinearRegressor(
    feature_columns=feature_cols,
    model_dir="./modeldir"
)
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

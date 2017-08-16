import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn import LinearRegressor
from tensorflow.contrib import learn
from tensorflow.contrib import layers
from tensorflow.contrib.learn.python.learn.utils import input_fn_utils

FEATURES = ["input_a", "input_b"]

def input_fn_train():
    feature_cols = {
        "input_a": tf.constant([[1], [2], [3]]),
        "input_b": tf.constant([[0], [-7], [4]])
    }
    outputs = tf.constant([-1, 0, 17])
    return feature_cols, outputs

def input_fn_test():
    feature_cols = {
        "input_a": tf.constant([[1], [2], [3]]),
        "input_b": tf.constant([[0], [-7], [4]])
    }
    return feature_cols

def model(features, targets, mode):
    W = tf.get_variable("W", [1, len(features)])
    b = tf.get_variable("b", [1])
    X = [features[key] for key in features]
    X = tf.cast(X, tf.float32)
    X = tf.reshape(X, [2, 3])
    y = tf.reshape(tf.matmul(W, X) + b, [-1])

    loss = tf.losses.mean_squared_error(targets, y)

    global_step = tf.train.get_global_step()
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))

    return tf.contrib.learn.ModelFnOps(
        mode=mode,
        predictions=y,
        loss=loss,
        train_op=train
    )

#feature_cols = [layers.real_valued_column(name) for name in FEATURES]
#regressor = LinearRegressor(feature_columns=feature_cols, model_dir="./modeldir")
regressor = learn.Estimator(model_fn=model, model_dir="./modeldir")
regressor.fit(input_fn=input_fn_train, steps=150)
pred = regressor.predict(input_fn=input_fn_test)
print [p for p in pred]

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

import tensorflow as tf
import pandas as pd

sess = tf.InteractiveSession() # can't call Tensor.eval() without this on

FEATURES = ["ttt30", "all", "MEP", "triaged", "PRIO3", "PRIO4"]

pdframe = pd.read_csv("NALState2017-07-11T15:14:43.994+02:00.csv")

def input_fn_train():
    feature_cols = {name: tf.constant(pdframe[name].get_values()[:-1]) for name in FEATURES}
    outputs = tf.constant(pdframe["ttt30"].get_values()[1:]) # predicting next ttt30 when sample intervals are 23 min, well well
    return feature_cols, outputs





import pandas as pd
import tensorflow as tf
import numpy as np


pdframe = pd.read_csv("QLasso2017-08-04T16:02:58.809+02:00.csv")
epochsecond = np.array(pdframe["epochseconds"].get_values(), dtype=np.int32)
ttl_next_low_prio_patient = np.array(pdframe["TTLOfNextPatient"].get_values(), dtype=np.float32)

WEEK_IN_SECONDS = 3600 * 24 * 7
BIN_WIDTH = 600 # in seconds
N_BINS = WEEK_IN_SECONDS / BIN_WIDTH

wait_time_bins = {i: [] for i in range(N_BINS)}
for i in range(len(epochsecond)):
    second_in_week = epochsecond[i] % WEEK_IN_SECONDS
    bin_number = second_in_week / BIN_WIDTH
    wait_time_bins[bin_number] = wait_time_bins[bin_number] + [ttl_next_low_prio_patient[i]]

def to_time_of_week_feature_np(epochsecond):
    wait_time_mean_bins = np.array([np.mean(wait_time_bins[i]) for i in range(N_BINS)], dtype=np.float32)
    second_in_week = epochsecond % WEEK_IN_SECONDS
    return wait_time_mean_bins[second_in_week / BIN_WIDTH]

def to_time_of_week_feature(epochsecond):
    wait_time_mean_bins = tf.constant([np.mean(wait_time_bins[i]) for i in range(N_BINS)], dtype=tf.float32)
    second_in_week = epochsecond % WEEK_IN_SECONDS
    return tf.map_fn(lambda s: wait_time_mean_bins[s / BIN_WIDTH], second_in_week, dtype=tf.float32)

if __name__ == "__main__":
    sess = tf.Session()

    pdframe = pd.read_csv("QLasso2017-08-04T16:02:58.809+02:00.csv")

    epochsecond = tf.constant(pdframe["epochseconds"].get_values(), dtype=tf.int32)
    ttl_next_low_prio_patient = tf.constant(pdframe["TTLOfNextPatient"].get_values(), dtype=tf.float32)
    pred = to_time_of_week_feature(epochsecond)
    mse_minutes = tf.losses.mean_squared_error(ttl_next_low_prio_patient, pred) / 3600.
    print sess.run(mse_minutes)

    epochsecond_np = np.array(pdframe["epochseconds"].get_values(), dtype=np.int32)
    ttl_next_low_prio_patient_np = np.array(pdframe["TTLOfNextPatient"].get_values(), dtype=np.float32)
    pred_np = to_time_of_week_feature_np(epochsecond_np)
    print np.mean((pred_np - ttl_next_low_prio_patient_np) ** 2) / 3600

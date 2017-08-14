# -*- coding: utf-8 -*-

import pandas as pd
import tensorflow as tf
import numpy as np


pdframe = pd.read_csv("FLasso2017-08-14T15:29:28.093+02:00.csv")

WEEK_IN_SECONDS = 3600 * 24 * 7
BIN_WIDTH = 600 # in seconds
N_BINS = WEEK_IN_SECONDS / BIN_WIDTH

def _incidence_bins(event_title):
    incidence_bins = {i: [] for i in range(N_BINS)}
    epochsecond = np.array(pdframe["epochseconds"].get_values(), dtype=np.int32)
    next_hour_incidence = np.array(pdframe["NextHour" + event_title].get_values(), dtype=np.float32)
    for i in range(len(epochsecond)):
        second_in_week = epochsecond[i] % WEEK_IN_SECONDS
        bin_number = second_in_week / BIN_WIDTH
        incidence_bins[bin_number] = incidence_bins[bin_number] + [next_hour_incidence[i]]
    return incidence_bins

def to_time_of_week_feature_np(epochsecond, event_title):
    incidence_bins = _incidence_bins(event_title)
    incidence_mean_bins = np.array([np.mean(incidence_bins[i]) for i in range(N_BINS)], dtype=np.float32)
    second_in_week = epochsecond % WEEK_IN_SECONDS
    return incidence_mean_bins[second_in_week / BIN_WIDTH]

def to_time_of_week_feature(epochsecond, event_title):
    incidence_bins = _incidence_bins(event_title)
    incidence_mean_bins = tf.constant([np.mean(incidence_bins[i]) for i in range(N_BINS)], dtype=tf.float32)
    second_in_week = epochsecond % WEEK_IN_SECONDS
    return tf.map_fn(lambda s: incidence_mean_bins[s / BIN_WIDTH], second_in_week, dtype=tf.float32)

if __name__ == "__main__":
    sess = tf.Session()

    pdframe = pd.read_csv("FLasso2017-08-14T15:29:28.093+02:00.csv")

    for event_title in ["Kölapp", "Triage", "Klar", "Läkare"]:
        print "time-of-week mse for NextHour" + event_title + ":"

        epochsecond = tf.constant(pdframe["epochseconds"].get_values(), dtype=tf.int32)
        next_hour_incidence = tf.constant(pdframe["NextHour" + event_title].get_values(), dtype=tf.float32)
        pred = to_time_of_week_feature(epochsecond, event_title)
        mse_minutes = tf.losses.mean_squared_error(next_hour_incidence, pred)
        print sess.run(mse_minutes)

        epochsecond_np = np.array(pdframe["epochseconds"].get_values(), dtype=np.int32)
        next_hour_incidence = np.array(pdframe["NextHour" + event_title].get_values(), dtype=np.float32)
        pred_np = to_time_of_week_feature_np(epochsecond_np, event_title)
        print np.mean((pred_np - next_hour_incidence) ** 2)

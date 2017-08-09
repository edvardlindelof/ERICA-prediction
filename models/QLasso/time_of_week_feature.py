import pandas as pd
import tensorflow as tf


def to_time_of_week_feature(epochsecond, wait_time):
    print epochsecond
    print wait_time
    WEEK_IN_SECONDS = 3600 * 24 * 7
    BIN_WIDTH = 600 # in seconds
    N_BINS = WEEK_IN_SECONDS / BIN_WIDTH

    second_in_week = epochsecond % WEEK_IN_SECONDS

    def to_bin(bin_number):
        elements_of_bin = tf.where(tf.equal(second_in_week / BIN_WIDTH, bin_number))
        wait_times_of_bin = tf.gather(wait_time, elements_of_bin)
        #wait_times_of_bin = ttl_next_low_prio_patient[tf.where(elements_of_bin > 0)]
        return tf.reduce_mean(wait_times_of_bin)

    elements_of_a_bin = tf.where(tf.equal(second_in_week / BIN_WIDTH, 0)) # [0,1,..,1,1,0]

    binned_wait_time_means = tf.map_fn(to_bin, tf.range(N_BINS), dtype=tf.float64)

    return tf.map_fn(lambda s: binned_wait_time_means[s / BIN_WIDTH], second_in_week, dtype=tf.float64)


if __name__ == "__main__":
    sess = tf.Session()

    pdframe = pd.read_csv("QLasso2017-08-04T16:02:58.809+02:00.csv")
    epochsecond = tf.constant(pdframe["epochseconds"].get_values(), dtype=tf.int32)
    ttl_next_low_prio_patient = tf.constant(pdframe["TTLOfNextPatient"].get_values(), dtype=tf.float64)

    pred = to_time_of_week_feature(epochsecond, ttl_next_low_prio_patient)
    mse_minutes = tf.losses.mean_squared_error(ttl_next_low_prio_patient, pred) / 3600.
    print sess.run(mse_minutes)

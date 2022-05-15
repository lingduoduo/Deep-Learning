import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gzip
import os
import shutil
from history import add_history, plot_history, save_history

os.environ["TF_CPP_LOG_LEVEL"] = "2"


def retrieve_data():
    cache_dir = ".."
    cache_subdir = "data"
    traffic_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz"
    traffic_compressed = tf.keras.utils.get_file("traffic.csv.gz", traffic_url, cache_dir=cache_dir, cache_subdir=cache_subdir)

    out_file = f"{cache_dir}/{cache_subdir}/traffic.csv"
    if not os.path.exists(out_file) and not os.path.isfile(out_file):
        with gzip.open(traffic_compressed, 'rb') as t_in:
            with open(out_file, "wb") as t_out:
                shutil.copyfileobj(t_in, t_out)
    return out_file


def plot_sequence(time, sequences, start=0, end=None):
    y_max = 1.0
    if len(np.shape(sequences)) == 1:
        sequences = [sequences]
    time = time[start:end]
    plt.figure(figsize=(28, 8))
    for sequence in sequences:
        y_max = max(np.max(sequence), y_max)
        sequence = sequence[start:end]
        plt.plot(time, sequence)
    plt.show()


if __name__ == "__main__":
    traffic_file = retrieve_data()

    # Load data
    traffic_df = pd.read_csv(traffic_file)
    traffic_volume = np.array(traffic_df["traffic_volume"])
    time_steps = np.array(list(range(len(traffic_volume))))

    # Extra variables for visualization
    day = 24
    week = 168
    month = 672

    traffic_df.head()
    traffic_df.describe()

    plot_sequence(time_steps, traffic_volume, end=week)
    plot_sequence(time_steps, traffic_volume, end=month)

    


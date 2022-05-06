import csv
import os
import urllib.request


def retrieve_url(url, filename):
    if not os.path.exists(filename) and not os.path.isfile(filename):
        urllib.request.urlretrieve(url, filename)
    else:
        print(f"{filename} already exists! Nothing to download")

if __name__ == "__main__":
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/slump/slump_test.data"
    data_path = 
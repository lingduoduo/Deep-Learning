import keras.callbacks
import json


def save_history(history, model_name):
    hist_out = {}
    hist_out["epoch"] = history.epoch
    hist_out["history"] = history.history
    hist_out["params"] = history.params
    with open(f"{model_name}.history", "w") as outfile:
        json.dump(hist_out, outfile)


def load_history(model_name, model_format=""):
    with open(f"{model_name}.history", "r") as f:
        hist = json.load(f)
    history = keras.callbacks.History()
    history.epoch = hist["epoch"]
    history.history = hist["history"]
    history.params = hist["params"]
    model = keras.models.load_model(f"{model_name}{model_format}")
    history.set_model(model)

    return history

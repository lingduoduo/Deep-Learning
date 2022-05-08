from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os


if __name__ == "__main__":
    url = "https://storage.googleapis.com/acg-datasets/fmist_test.tgz"
    cache_dir = ".."
    cache_subdir = "data"
    keras.utils.get_file("fmnist_test.tgz", url, extract=True, cache_dir=cache_dir, cache_subdir=cache_subdir)

    extract_path = f"{cache_dir}/{cache_subdir}/fmnist_test"
    class_zero_images = os.listdir(f"{extract_path}/0") ## directory with images
    im = Image.open(f"{extract_path}/0/{class_zero_images[0]}")
    plt.imshow(im, cmap="Greys")
    plt.show()

    test_preprocess = keras.preprocessing.images.ImageDataGenerator(rescale=1.0/255)

    test_generator = test_preprocess.flow_from_directory(
        extract_path,
        target_size=(28, 28),
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=2,
        shuffle=True,

    )

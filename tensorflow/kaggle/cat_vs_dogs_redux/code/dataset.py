from PIL import Image
import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import layers
from tensorflow.python.ops.image_ops_impl import ResizeMethod
import matplotlib.pyplot as plt


def get_images_path(path: str) -> [str]:
    image_path_array = os.listdir(path)
    return [path + item for item in image_path_array]


def preprocess(path: str):

    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(images=image, size=[150, 150], method=ResizeMethod.BICUBIC)
    image = tf.cast(image, dtype=tf.uint8)
    #TODO: NOT WORKING. LABEL DOG AND CAT CLASS
    label = tf.strings.regex_full_match(path, "dog")

    return image, label


def tf_dataset(path: str, batch_size: int = 8, buffer_size: int = 1000, shuffle: bool = True):
    images_path = get_images_path(path)
    dataset = tf.data.Dataset.from_tensor_slices(images_path)
    dataset = dataset.map(preprocess).batch(batch_size=batch_size)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size)

    return dataset


def main():
    dataset = tf_dataset('../data/train/')

    for x,y in dataset:
        print(x.numpy()[0])
        plt.imsave('img.jpg', x.numpy()[0])
        print(y.numpy())
        break
    model = keras.Sequential(
        [
            layers.Input((28, 28, 3)),
            layers.Conv2D(16, 3, padding="same"),
            layers.Conv2D(32, 3, padding="same"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(2),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=[keras.losses.SparseCategoricalCrossentropy(from_logits=True)],
        metrics=["accuracy"]
    )

    model.fit(dataset, epochs=10, verbose=2)


if __name__ == "__main__":
    main()

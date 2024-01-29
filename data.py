from config import batch_size, image_shape, epochs, validation_split, num_classes, data_folder, classes_in
from keras.utils import image_dataset_from_directory
from os import listdir, path, remove
from tensorflow import Tensor
from typing import Tuple
import tensorflow as tf
import pathlib
import random


def preprocess_data(image, label) -> Tuple[Tensor, Tensor]:
    image = preprocess_images(image)
    label = preprocess_label(label)
    return image, label


def preprocess_images(images) -> Tensor:
    images = tf.image.resize_with_pad(images, target_height=image_shape, target_width=image_shape)
    images = tf.image.random_flip_left_right(images)
    return images


def preprocess_label(labels) -> Tensor:
    mask0 = tf.greater(labels, num_classes - 1)
    modified_tensor = tf.where(mask0, -1 * tf.ones_like(labels), labels)
    return tf.one_hot(modified_tensor, depth=num_classes)


def prepare_dataset(subset='training'):
    data_dir = pathlib.Path(f'{data_folder}/')
    ds = image_dataset_from_directory(
        data_dir,
        class_names=classes_in,
        validation_split=validation_split,
        subset=subset,
        seed=123,
        batch_size=batch_size,
        shuffle=True)

    pipeline = (tf.data.Dataset.range(epochs)
    .interleave(lambda _: ds, num_parallel_calls=2)
    .map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE).prefetch(
        tf.data.AUTOTUNE)
    )
    return pipeline


def add_random_black_space(image: Tensor) -> Tensor:
    pad_size = random.randint(0, 81)
    paddings = tf.constant(
        [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]])
    image = tf.pad(image, paddings, "CONSTANT")
    return image


def random_flip_batch(batch_images):
    flip_indices = tf.random.uniform(shape=[tf.shape(batch_images)[0]]) < 0.5
    flipped_images = tf.where(
        flip_indices, tf.image.flip_left_right(batch_images), batch_images)
    return flipped_images


def clean_last_train(dirs=None):
    if dirs is None:
        dirs = ['models']
    for directory in dirs:
        for f in listdir(directory):
            remove(path.join(directory, f))

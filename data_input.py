import numpy as np
import tensorflow as tf
import hdf5storage as hdf
import param

num_labels = param.num_labels


def decode(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features={
            'train/data': tf.FixedLenFeature([], tf.string),
            'train/label': tf.FixedLenFeature([], tf.int64),
        }
    )
    image = tf.decode_raw(features['train/data'], tf.uint8)
    label = tf.cast(features['train/label'], tf.int32)

    image = tf.reshape(image, [300, 300, 1])
    return image, label


def normalize(image, label):
    image = tf.cast(image, dtype=tf.float32) * (1. / 255)
    return image, label


def reformat(image, label):
    label = tf.one_hot(indices=label, depth=num_labels)
    return image, label


def data_iterator(num_epochs, batch_size, tf_filename):
    with tf.name_scope('data_input'):
        dataset = tf.data.TFRecordDataset(tf_filename)
        dataset = dataset.map(decode)
        dataset = dataset.map(normalize)
        dataset = dataset.map(reformat)
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
    return iterator


def load_data(filename):
    """

    :param filename:
    :return:
    """
    images = hdf.loadmat(filename[0])
    images = images[filename[0][0:-4]]
    images = np.float32(np.reshape(images, [images.shape[0], images.shape[1], images.shape[2], 1]))

    labels = hdf.loadmat(filename[1])
    labels = np.float32(labels[filename[1][0:-4]])
    labels = np.reshape(labels, [labels.shape[0], ])
    labels = tf.one_hot(labels, num_labels)

    return images, labels



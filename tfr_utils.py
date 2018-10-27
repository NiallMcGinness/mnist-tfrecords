from __future__ import print_function
import os
import tensorflow as tf
import numpy as np


def tfr_write(save_dir,data_sets):
    data_splits = ["train", "test", "validation"]
    for d in range(len(data_splits)):
        data_set = data_sets[d]
        filename = os.path.join(save_dir, data_splits[d] + '.tfrecords')
        writer = tf.python_io.TFRecordWriter(filename)
        for index in range(data_set.images.shape[0]):
            # need to call .tostring() to convert numpy array
            # to bytes 
            image = data_set.images[index].tostring()
            example = tf.train.Example(features=tf.train.Features(
                feature={
                    'height': tf.train.Feature(
                        int64_list=tf.train.Int64List(
                            value=[data_set.images.shape[1]])),
                    'width': tf.train.Feature(
                        int64_list=tf.train.Int64List(
                            value=[data_set.images.shape[2]])),
                    'depth': tf.train.Feature(
                        int64_list=tf.train.Int64List(
                            value=[data_set.images.shape[3]])),
                    'label': tf.train.Feature(
                        int64_list=tf.train.Int64List(
                            value=[int(data_set.labels[index])])),
                    'image_raw': tf.train.Feature(
                        bytes_list=tf.train.BytesList(
                            value=[image]))}))
            writer.write(example.SerializeToString())
        writer.close()
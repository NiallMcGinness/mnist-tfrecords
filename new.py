import tensorflow as tf
import numpy as np
import os, sys
import time
import logging 

#feature_column = [tf.feature_column.numeric_column(key='image', shape=(784))]

feature_column = [tf.feature_column.numeric_column(key="image", shape=(784,))]

model = tf.estimator.DNNClassifier([100,100], n_classes=10, feature_columns=feature_column)
logging.getLogger().setLevel(logging.INFO)
def _parse_(serialized_example):
    feature = {'image_raw':tf.FixedLenFeature([],tf.string),
                'label':tf.FixedLenFeature([],tf.int64)}
    example = tf.parse_single_example(serialized_example,feature)
    image = tf.decode_raw(example['image_raw'],tf.uint8) #remember to parse in int64. float will raise error
    label = tf.cast(example['label'],tf.int32)
    return (dict({'image':image}),label)

mnist_tfrecord_path = os.getcwd() + "/data/mnist/train.tfrecords"

def tfrecord_train_input_fn(batch_size=32):
    tfrecord_dataset = tf.data.TFRecordDataset(mnist_tfrecord_path)
    tfrecord_dataset = tfrecord_dataset.map(lambda   x:_parse_(x)).shuffle(True).batch(batch_size)
    tfrecord_iterator = tfrecord_dataset.make_one_shot_iterator()
    return tfrecord_iterator.get_next()

model.train(lambda:tfrecord_train_input_fn(32),steps=2000)
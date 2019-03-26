import tensorflow as tf
import numpy as np
import os, sys
import time

train_tfrecord_path = os.getcwd() + "/data/mnist/train.tfrecords"

test_tfrecord_path = os.getcwd() + "/data/mnist/test.tfrecords"

def parser(record):
    keys_to_features = {
        "image_raw": tf.FixedLenFeature([], tf.string),
        "label":     tf.FixedLenFeature([], tf.int64)
    }
    parsed = tf.parse_single_example(record, keys_to_features)
    image = tf.decode_raw(parsed["image_raw"], tf.uint8)
    image = tf.cast(image, tf.float32)
    #image = tf.reshape(image, shape=[224, 224, 3])
    label = tf.cast(parsed["label"], tf.int32)

    return {'image': image}, label

def input_fn(filenames, batch_size=32, epochs=2):
  dataset = tf.data.TFRecordDataset(filenames=filenames, num_parallel_reads=40)
  """
  dataset = dataset.apply(
      tf.contrib.data.shuffle_and_repeat(1024, 1)
  )
  dataset = dataset.apply(
      tf.contrib.data.map_and_batch(parser, 32)
  )
  """
  dataset = dataset.map(parser, num_parallel_calls=12)
  dataset = dataset.shuffle(True).repeat(epochs)
  dataset = dataset.batch(batch_size)
  #dataset = dataset.
  #
  #dataset = dat11aset.batch(batch_size=1000)
  #dataset = dataset.prefetch(buffer_size=2)
  ds_iter = dataset.make_one_shot_iterator()
  ds_iter = ds_iter.get_next()
  return ds_iter


#train_data_set = input_fn(train_tfrecord_path)

feature_column = [tf.feature_column.numeric_column(key="image", shape=(784,))]

model = tf.estimator.DNNClassifier([100,100], n_classes=10, feature_columns=feature_column)
#lambda:tfrecord_train_input_fn(32)

count = 0 
while ( count < 100 ):
	model.train(lambda:input_fn(train_tfrecord_path), steps=1000)
	result = mode.evaluate(lambda:input_fn(test_tfrecord_path))
	print(result)
	print(" accuracy : {} ".format(result["accuracy"]))
	sys.stdout.flush()
	count += 1
from __future__ import print_function
import os
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import mnist
import numpy as np
from tfr_utils import tfr_write
from nn import cnn

save_dir = os.getcwd() + "/data/mnist/"

data_sets = mnist.read_data_sets(save_dir,
                                 dtype=tf.uint8,
                                 reshape=False,
                                 validation_size=1000)

tfr_write(save_dir, data_sets)


filename = os.path.join(save_dir, 'train.tfrecords')
record_iterator = tf.python_io.tf_record_iterator(filename)


num_epochs = 10

#number of epochs will be fed into the queue, this just multiplies the number 
#of items in the queue by the epoch number 

filename_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)
reader = tf.TFRecordReader()
_ , serialized_example = reader.read(filename_queue)

features = tf.parse_single_example(serialized_example, 
                                   features={
                                       'image_raw' : tf.FixedLenFeature([],tf.string),
                                       'label' : tf.FixedLenFeature([], tf.int64)
                                   })

image = tf.decode_raw(features['image_raw'], tf.uint8)
# height and width of image is 28 pixels,  28*28 = 784
image.set_shape([784])

image = tf.cast(image, tf.float32) * (1. / 255 ) 
label  = tf.cast(features['label'], tf.int32 )

images_batch, labels_batch = tf.train.shuffle_batch([image,label],
                                                     batch_size=128,
                                                     capacity=2000,
                                                     min_after_dequeue=1000)

# keep probability just the inverse of 'dropout' rate used in some other frameworks 
# keep out probability of 0.75 == dropout rate of 0.25  
keep_prob = 0.75
y_pred  = cnn(images_batch, keep_prob)



loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred,labels=labels_batch)
loss_mean = tf.reduce_mean(loss)
train_op = tf.train.AdamOptimizer().minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

init = tf.local_variables_initializer()
sess.run(init)

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess,coord=coord)


try:
    step = 0
    while not coord.should_stop():
        step += 1
        sess.run([train_op])
        if step % 100 == 0 :
            loss_mean_val =  sess.run([loss_mean])
            print(" step# {} , loss_mean_val {} ".format(step,loss_mean_val) )
except tf.errors.OutOfRangeError:
    print(" done training {} epochs , {} steps ".format(num_epochs, step))
finally:
    coord.request_stop()

coord.join(threads)
sess.close()


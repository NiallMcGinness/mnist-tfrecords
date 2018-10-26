import tensorflow as tf


def cnn(images_batch, keep_prob):
    print(" images_batch shape : {} ".format(images_batch.shape))
    
    with tf.name_scope('reshape'):
        x_image = tf.reshape(images_batch, [-1, 28, 28, 1])
        print("  x_image.shape: {} ".format(x_image.shape))
    # this fist convolution builds a graph with 32 filters
    # the graph and filter sizes can be changed and experimented with pretty arbitrarily 
    # but the nth value of a tensor must match the n-1 th value of the other tensor you are multiplying
    # here the value is 1 as this is the colour channel of the mnist images ( they are graysacale ), if you were building out 
    # a graph with 3 colour channels the next tensor in the graph would need to have 3 as its n-1 th property 
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([6, 6, 1, 32])
        print("  W_conv1.shape: {} ".format(W_conv1.shape))
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        print("  h_conv1.shape: {} ".format(h_conv1.shape))
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)
        print("  h_pool1.shape: {} ".format(h_pool1.shape))
    # the second convolution takes the 32 filters built in the previous step
    # a builds a graph with 64 filters 
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        print("  W_conv2.shape: {} ".format(W_conv2.shape))
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        print("  h_conv2.shape: {} ".format(h_conv2.shape))

    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)
        print("  h_pool2.shape: {} ".format(h_pool2.shape))
   
    feature_number = 1024
    # after convolving over the images and building filters we now
    # flatten them out into 2 fully connected layers 
    # with a drop out layer in between them to avoid overfitting
    # the feature number defined marks the last step before 
    # reducing the output of the whole graph to the number of classes we are modeling
    # in mnist this is 10 for the ten digits ( 0 - 9 ) we are classifying
    # you can vary this to experiment  
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([7 * 7 * 64, feature_number])
        print("  W_fc1.shape: {} ".format(W_fc1.shape))
        b_fc1 = bias_variable([feature_number])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        print("  h_pool2_flat.shape: {} ".format(h_pool2_flat.shape))
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    with tf.name_scope('dropout'):
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        print("  h_fc1_drop .shape: {} ".format(h_fc1_drop.shape))

    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([feature_number, 10])
        b_fc2 = bias_variable([10])
        print("  W_fc2.shape: {} ".format(W_fc2.shape))
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        print("  y_conv.shape: {} ".format(y_conv.shape))

    return y_conv


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
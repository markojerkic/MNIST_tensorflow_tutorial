import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import numpy as np

graph = tf.Graph()

batch_size = 128

with graph.as_default():

    def generate_weights_biases(weights_shape, biases_shape, name):
        with tf.name_scope("generate_{}".format(name)):
            # Create random weights
            weights = tf.Variable(tf.truncated_normal(weights_shape, stddev=0.1), name="weights_{}".format(name))
            # All biases are set zeros in the beginning
            biases = tf.Variable(tf.zeros(biases_shape), name="biases_{}".format(name))
            return weights, biases

    with tf.name_scope("data"):
        # Training data
        data = tf.placeholder(tf.float32, [batch_size, 28, 28, 1], name="train_data")
        labels = tf.placeholder(tf.float32, [batch_size, 10], name="train_labels")
        # Test data
        test_data = tf.placeholder(tf.float32, [batch_size, 28, 28, 1], name="test_data")
        test_labels = tf.placeholder(tf.float32, [batch_size, 10], name="test_labels")
        # Keep probability for the dropout
        keep_prob = tf.placeholder(tf.float32, name="keep_prob")

    with tf.name_scope("conv_layer_1"):
        weights_conv1, biases_conv1 = generate_weights_biases([5, 5, 1, 16], [16], "conv1")
        # First convolutional layer for the training dataset
        logits = tf.nn.conv2d(data, weights_conv1, strides=[1, 1, 1, 1], padding='SAME') + biases_conv1
        logits = tf.nn.relu(logits)
        # First convolutional layer for the test dataset
        test_logits = tf.nn.conv2d(test_data, weights_conv1, strides=[1, 1, 1, 1], padding='SAME') + \
                      biases_conv1
        test_logits = tf.nn.relu(test_logits)
        with tf.name_scope("max_pool_1"):
            # Pooling the training logits
            logits = tf.nn.max_pool(logits, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1], padding='SAME')
            # Pooling the test logits
            test_logits = tf.nn.max_pool(test_logits, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1], padding='SAME')

    with tf.name_scope("conv_layer_2"):
        weights_conv2, biases_conv2 = generate_weights_biases([5, 5, 16, 16], [16], "conv2")
        # Second convolutional layer for the training data
        logits = tf.nn.conv2d(logits, weights_conv2, strides=[1, 1, 1, 1], padding='SAME') + biases_conv2
        logits = tf.nn.relu(logits)
        # Second convolutional layer fot the test data
        test_logits = tf.nn.conv2d(test_logits, weights_conv2, strides=[1, 1, 1, 1], padding='SAME') + \
                      biases_conv2
        test_logits = tf.nn.relu(test_logits)
        with tf.name_scope("max_pool_2"):
            # Pooling for training and testing data
            logits = tf.nn.max_pool(logits, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1], padding='SAME')
            test_logits = tf.nn.max_pool(test_logits, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1], padding='SAME')

    with tf.name_scope("connected_1"):
        weights_1, biases_1 = generate_weights_biases([7*7*16, 1024], [1024], "fully1")
        # Reshape the training logits to be a two-dimensional tensor
        logits = tf.reshape(logits, [-1, 7*7*16])
        # Perform matrix multiplication and add a relu layer
        logits = tf.nn.relu(tf.matmul(logits, weights_1) + biases_1, name="relu1")
        # Apply dropout
        logits = tf.nn.dropout(logits, keep_prob=keep_prob)
        # Repeat everything for the test data
        test_logits = tf.reshape(test_logits, [-1, 7*7*16])
        test_logits = tf.nn.relu(tf.matmul(test_logits, weights_1) + biases_1, name="relu1")

    with tf.name_scope("connected_2"):
        weights_2, biases_2 = generate_weights_biases([1024, 10], [10], "fully2")
        # The second fully connected layer is the last layer, so
        # we do not need to add dropout or a relu layer
        logits = tf.matmul(logits, weights_2) + biases_2
        # Final test dataset matrix multiplication
        test_logits = tf.matmul(test_logits, weights_2) + biases_2

    with tf.name_scope("loss"):
        # Compute the softmax loss
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    tf.summary.scalar("loss", loss)

    with tf.name_scope("optimizer"):
        optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    with tf.name_scope("accuracy"):
        def calculate_accuracy(log, lab):
            # Calculate the accuracy of our prediction
            # The accuracy is expressed in percentages
            correct_prediction = tf.equal(tf.argmax(log, 1), tf.argmax(lab, 1))
            return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        accuracy = calculate_accuracy(logits, labels)
        test_accuracy = calculate_accuracy(test_logits, test_labels)
    tf.summary.scalar("accuracy", accuracy)
    tf.summary.scalar("test_accuracy", test_accuracy)
    merged = tf.summary.merge_all()


steps = 2001

with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()
    writer = tf.summary.FileWriter("/tmp/summary/2", tf.get_default_graph())
    print('Initialized!')
    for step in range(steps):
        # Retrieve a batch of MNIST images
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = np.reshape(batch_x, [-1, 28, 28, 1])
        # Get a batch of test data
        test_x, test_y = mnist.test.next_batch(batch_size)
        test_x = np.reshape(test_x, [-1, 28, 28, 1])
        # Feed dictionary for the placeholders
        feed_dict = {data: batch_x, labels: batch_y, test_data: test_x,
                     test_labels: test_y, keep_prob: 0.5}

        _, a, ta, l, m = sess.run([optimizer, accuracy, test_accuracy, loss, merged], feed_dict=feed_dict)

        # Write to Tensorboard
        writer.add_summary(m, step)

        # Print accuracy and loss
        if step % 500 == 0:
            print("Accuracy at step {}: {}".format(step, a))
            print("     loss: {}".format(l))
            print("         test accuracy: {}".format(ta))
    # Close the writer
    writer.close()

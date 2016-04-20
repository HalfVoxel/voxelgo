import tensorflow as tf
import math


def weigth_var(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_var(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


class Net:
    def __init__(self):
        size = 19
        pixels = size*size
        input_size = pixels
        input_channels = 5
        output_size = pixels

        self.x = tf.placeholder(tf.float32, [None, input_size*input_channels], name="x")

        # One hot vector with a 1 for the action that was actually taken
        self.action_taken = tf.placeholder(tf.float32, [None, output_size], name="action_taken")
        # Max Q of the state after the action was taken
        self.next_q = tf.placeholder(tf.float32, [None, 1], name="next_q")

        self.W_conv1 = weigth_var([5, 5, input_channels, 64], "W_conv1")
        self.b_conv1 = bias_var([64], "b_conv1")
        x_image = tf.reshape(self.x, [-1, size, size, input_channels])

        # Shape [?, size, size, 32]
        h_conv1 = tf.nn.relu(conv2d(x_image, self.W_conv1) + self.b_conv1)
        self.h_conv1 = h_conv1

        # self.W_conv2 = weigth_var([5, 5, 64, 32], "W_conv2")
        # self.b_conv2 = bias_var([32], "b_conv2")
        # h_conv2 = tf.nn.relu(conv2d(h_conv1, self.W_conv2) + self.b_conv2)

        self.W_conv3 = weigth_var([5, 5, 64, 8], "W_conv3")
        self.b_conv3 = bias_var([8], "b_conv3")
        h_conv3 = tf.nn.relu(conv2d(h_conv1, self.W_conv3) + self.b_conv3)

        self.W_conv4 = weigth_var([5, 5, 8, 1], "W_conv4")
        self.b_conv4 = bias_var([1], "b_conv4")
        h_conv4 = conv2d(h_conv3, self.W_conv4) + self.b_conv4

        h_conv4_flat = tf.reshape(h_conv4, [-1, size*size])

        # Output
        self.y = h_conv4_flat
        self.y_taken = tf.reduce_sum(self.action_taken * self.y, 1)

        self.max_q = tf.reduce_max(self.y, 1)
        self.move = tf.argmax(self.y, 1)

        # Loss
        self.loss = tf.reduce_mean(tf.square(self.y_taken - self.next_q))

        self.train_step = tf.train.AdamOptimizer(1.0).minimize(self.loss)

    def moves(self, sess, batch_xs):
        moves_flat = sess.run(self.move, feed_dict={self.x: batch_xs})
        return [(p % 19, p // 19) for p in moves_flat]

    def max_qs(self, sess, batch_xs):
        return sess.run(self.max_q, feed_dict={self.x: batch_xs})

    def q_taken(self, sess, batch_xs, action_taken):
        return sess.run(self.y_taken, feed_dict={self.x: batch_xs, self.action_taken: action_taken})

    def scores(self, sess, batch_xs):
        return sess.run(self.y, feed_dict={self.x: batch_xs})

    def loss(self, sess, batch_xs, action_taken, next_q):
        feed_dict = {self.x: batch_xs, self.action_taken: action_taken, self.next_q: next_q}
        loss = math.sqrt(sess.run(self.loss, feed_dict=feed_dict))
        return loss

    def train(self, sess, batch_xs, action_taken, next_q):
        feed_dict = {self.x: batch_xs, self.action_taken: action_taken, self.next_q: next_q}
        _, loss = sess.run([self.train_step, self.loss], feed_dict=feed_dict)
        return math.sqrt(loss)

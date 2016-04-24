import tensorflow as tf
import math
import numpy

def weigth_var(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_var(shape, name):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial, name=name)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


class Net:
    SCORE_SCALE = 1  # 1.0 / (19 * 19)

    def __init__(self):
        size = 19
        pixels = size * size
        input_size = pixels
        input_channels = 5
        output_size = pixels

        self.x = tf.placeholder(tf.float32, [None, input_size * input_channels], name="x")
        # Normalize scores to [-1, 1]... wait, no normalization required here...!?!?
        x_scaled = self.x
        x_image = tf.reshape(x_scaled, [-1, size, size, input_channels])

        # One hot vector with a 1 for the action that was actually taken
        self.action_taken = tf.placeholder(tf.float32, [None, output_size], name="action_taken")
        # Max Q of the state after the action was taken
        self.next_q = tf.placeholder(tf.float32, [None], name="next_q")
        self.next_q_scaled = tf.reshape(self.next_q, [-1]) * Net.SCORE_SCALE

        W_conv1 = weigth_var([5, 5, input_channels, 64], "W_conv1")
        b_conv1 = bias_var([64], "b_conv1")
        h_conv1 = tf.nn.tanh(conv2d(x_image, W_conv1) + b_conv1)

        # W_conv2 = weigth_var([5, 5, 64, 32], "W_conv2")
        # b_conv2 = bias_var([32], "b_conv2")
        # h_conv2 = tf.nn.tanh(conv2d(h_conv1, W_conv2) + b_conv2)

        W_conv3 = weigth_var([5, 5, 64, 8], "W_conv3")
        b_conv3 = bias_var([8], "b_conv3")
        h_conv3 = tf.nn.tanh(conv2d(h_conv1, W_conv3) + b_conv3)

        W_conv4 = weigth_var([5, 5, 8, 1], "W_conv4")
        b_conv4 = bias_var([1], "b_conv4")
        h_conv4 = conv2d(h_conv3, W_conv4) + b_conv4

        h_conv4_flat = tf.reshape(h_conv4, [-1, size * size])

        # Output
        self.y = h_conv4_flat
        y_taken_scaled = tf.reduce_sum(self.action_taken * self.y, 1)
        # y_taken_scaled = tf.tanh(tf.reduce_sum(x_image * [1, -1, 0, 0, 0], [1, 2, 3]) + 0.001 * b_conv4)
        self.y_taken = y_taken_scaled / Net.SCORE_SCALE

        max_q_scaled = tf.reduce_max(self.y, 1)
        self.max_q = max_q_scaled / Net.SCORE_SCALE
        self.move = tf.argmax(self.y, 1)

        # Loss
        self.deltas = y_taken_scaled - self.next_q_scaled
        loss_scaled = tf.reduce_mean(tf.square(self.deltas))
        self.loss = loss_scaled / (Net.SCORE_SCALE * Net.SCORE_SCALE)

        self.train_step = tf.train.AdamOptimizer(0.001).minimize(loss_scaled)

    def moves(self, sess, batch_xs):
        moves_flat = sess.run(self.move, feed_dict={self.x: batch_xs})
        return [(p % 19, p // 19) for p in moves_flat]

    def max_qs(self, sess, batch_xs):
        return sess.run(self.max_q, feed_dict={self.x: batch_xs})

    def q_taken(self, sess, batch_xs, action_taken):
        return sess.run(self.y_taken, feed_dict={self.x: batch_xs, self.action_taken: action_taken})

    def scores(self, sess, batch_xs):
        return sess.run(self.y, feed_dict={self.x: batch_xs})

    def eval_loss(self, sess, batch_xs, action_taken, next_q):
        feed_dict = {self.x: batch_xs, self.action_taken: action_taken, self.next_q: next_q}
        loss = sess.run(self.loss, feed_dict=feed_dict)
        return math.sqrt(loss)

    def train(self, sess, batch_xs, action_taken, next_q):
        # scores = []
        # for xs in batch_xs:
        #     score = 0
        #     for i in range(0, len(xs), 5):
        #         score += xs[i + 0] - xs[i + 1]

        #     scores.append(score)
        feed_dict = {self.x: batch_xs, self.action_taken: action_taken, self.next_q: next_q}
        _, loss = sess.run([self.train_step, self.loss], feed_dict=feed_dict)
        # print("Q: " + str(c_q))
        # print("T: " + str(taken))
        # assert(len(taken) == len(scores))
        # assert(isinstance(taken[0], numpy.float32))
        # print(list(zip(taken, scores, next_q)))
        # deltas = [abs(x[0] - x[1]) for x in zip(taken, next_q)]
        # print(c_deltas)
        # print(list(zip(deltas, c_deltas)))
        # print(sum([x * x for x in deltas]))
        return math.sqrt(loss)

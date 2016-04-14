import tensorflow as tf
import goinput
import goutil
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
		self.y_ = tf.placeholder(tf.float32, [None, output_size], name="y_")



		self.W_conv1 = weigth_var([5, 5, input_channels, 64], "W_conv1")
		self.b_conv1 = bias_var([64], "b_conv1")
		x_image = tf.reshape(self.x, [-1, size, size, input_channels])

		# Shape [?, size, size, 32]
		h_conv1 = tf.nn.relu(conv2d(x_image, self.W_conv1) + self.b_conv1)
		self.h_conv1 = h_conv1

		self.W_conv2 = weigth_var([5, 5, 64, 32], "W_conv2")
		self.b_conv2 = bias_var([32], "b_conv2")
		h_conv2 = tf.nn.relu(conv2d(h_conv1, self.W_conv2) + self.b_conv2)

		self.W_conv3 = weigth_var([5, 5, 32, 8], "W_conv3")
		self.b_conv3 = bias_var([8], "b_conv3")
		h_conv3 = tf.nn.relu(conv2d(h_conv2, self.W_conv3) + self.b_conv3)

		# h_pool2 = max_pool_2x2(h_conv2)

		# W_fc1 = weigth_var([7 * 7 * 64, 1024])
		# b_fc1 = bias_var([1024])

		# h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
		# h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

		# keep_prob = tf.placeholder(tf.float32)
		# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

		self.W_conv4 = weigth_var([5, 5, 8, 1], "W_conv4")
		self.b_conv4 = bias_var([1], "b_conv4")
		h_conv4 = tf.nn.relu(conv2d(h_conv3, self.W_conv4) + self.b_conv4)

		h_conv4_flat = tf.reshape(h_conv4, [-1, size*size])
		self.y = tf.nn.softmax(h_conv4_flat)
		# Flatten
		# h_conv3_flat = tf.reshape(h_conv3, [-1, size*size*4])

		# Fully connected layer #1
		# W_fc1 = weigth_var([size*size*4, size*size])
		# b_fc1 = bias_var([size*size])

		# y = tf.nn.softmax(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

		self.cross_entropy = -tf.reduce_sum(self.y_*tf.log(self.y))

		self.train_step = tf.train.AdamOptimizer(0.002).minimize(self.cross_entropy)


		# Multiply the scores by random values
		rnd = tf.random_uniform(shape=tf.shape(self.y), name="random")
		weighted_y = self.y * rnd
		# Find the maximum score and output a list where all elements are zero
		# except for the elements (probably 1) which contained the maximum value previously
		random_sample = tf.less_equal(tf.reduce_max(weighted_y, 1), weighted_y)
		
		self.random_logscore = -tf.reduce_sum(tf.to_float(random_sample) * tf.log(self.y))

		#train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
		#correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
		#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	def scores(self, sess, batch_xs):
		return sess.run(self.y, feed_dict={self.x: batch_xs})
		
	def loss(self, sess, batch_xs, batch_ys):
		loss = sess.run(self.cross_entropy, feed_dict={self.x: batch_xs, self.y_: batch_ys}) / len(batch_xs)
		return loss

	def train(self, sess, batch_xs, batch_ys):
		sess.run(self.train_step, feed_dict={self.x: batch_xs, self.y_: batch_ys})
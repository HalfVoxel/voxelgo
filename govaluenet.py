import tensorflow as tf
import goinput
import goutil
import math

def weigth_var(shape, name):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial, name=name)

def bias_var(shape, name, initial=0.1):
	initial = tf.constant(initial, shape=shape)
	return tf.Variable(initial, name=name)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_3x3(x):
	return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='SAME')

class Net:
	def __init__(self):
		size = 19
		pixels = size*size
		input_size = pixels
		input_channels = 5
		output_size = 1

		self.x = tf.placeholder(tf.float32, [None, input_size*input_channels], name="x")
		self.y_ = tf.placeholder(tf.float32, [None, output_size], name="y_")


		# Layer 1
		self.W_conv1 = weigth_var([5, 5, input_channels, 64], "W_conv1")
		self.b_conv1 = bias_var([64], "b_conv1")
		x_image = tf.reshape(self.x, [-1, size, size, input_channels])

		# Layer2
		h_conv1 = tf.nn.relu(conv2d(x_image, self.W_conv1) + self.b_conv1)
		self.h_conv1 = h_conv1

		self.W_conv2 = weigth_var([5, 5, 64, 16], "W_conv2")
		self.b_conv2 = bias_var([16], "b_conv2")
		h_conv2 = tf.nn.relu(conv2d(h_conv1, self.W_conv2) + self.b_conv2)

		# Layer 3
		h_pool1 = max_pool_3x3(h_conv2)

		# Layer 4
		self.W_conv3 = weigth_var([5, 5, 16, 4], "W_conv3")
		self.b_conv3 = bias_var([4], "b_conv3")
		h_conv3 = tf.nn.relu(conv2d(h_pool1, self.W_conv3) + self.b_conv3)

		# Layer 6
		self.W_conv4 = weigth_var([5, 5, 4, 1], "W_conv4")
		self.b_conv4 = bias_var([1], "b_conv4", 0.0)
		h_conv4 = conv2d(h_conv3, self.W_conv4) + self.b_conv4

		# h_conv4_flat = tf.reshape(h_conv4, [-1, 7*7])
		# Layer 7
		self.y = tf.tanh(tf.reduce_sum(h_conv4, [1, 2, 3]))

		#self.cross_entropy = 9 - tf.reduce_mean(tf.square(self.y_ + 2*self.y))
		self.cross_entropy = -tf.reduce_mean(tf.log(self.y_*self.y + 1 + 0.0001) - math.log(2) + self.y_*self.y*0.4 - 0.4)

		self.train_step = tf.train.AdamOptimizer(0.0001).minimize(self.cross_entropy)
		
	def loss(self, sess, batch_xs, batch_ys):
		y = sess.run(self.y, feed_dict={self.x: batch_xs, self.y_: batch_ys})
		print(y[0])
		print(y)
		print(batch_ys)
		loss = sess.run(self.cross_entropy, feed_dict={self.x: batch_xs, self.y_: batch_ys})
		return loss

	def train(self, sess, batch_xs, batch_ys):
		sess.run(self.train_step, feed_dict={self.x: batch_xs, self.y_: batch_ys})
import tensorflow as tf
import goinput
import time
import argparse
import goutil
import sys

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
		#train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
		#correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
		#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	def scores(self, sess, batch_xs, batch_ys):
		return sess.run(self.y, feed_dict={self.x: batch_xs, self.y_: batch_ys})
		
	def loss(self, sess, batch_xs, batch_ys):
		loss = sess.run(self.cross_entropy, feed_dict={self.x: batch_xs, self.y_: batch_ys}) / len(batch_xs)
		return loss

	def train(self, sess, batch_xs, batch_ys):
		sess.run(self.train_step, feed_dict={self.x: batch_xs, self.y_: batch_ys})

parser = argparse.ArgumentParser(description="Go Neural Network")
parser.add_argument('--checkpoint', dest="checkpoint", default=None, help="path to checkpoint file")
parser.add_argument('--simulate', dest="simulate", action='store_true', help="simulate random games")
parser.add_argument('--train', dest="train", action='store_true', help="train the net")
parser.add_argument('--dump', dest="dump", action='store_true', help="dump training parameters to output")
parser.add_argument('--visualize', dest="visualize", action='store_true', help="visualize probabilities from stdin")

args = parser.parse_args()

net = Net()
saver = tf.train.Saver()
save_every = 500

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

def write_tensor(f, variable):
	name = variable.name.replace(":0","") + "_data"
	tensor = sess.run(variable)
	f.write("std::string " + name + " = ")
	f.write('"')
	f.write(str(tensor.ndim) + ' ')
	items = 1
	for i in range(0, tensor.ndim):
		items *= tensor.shape[i]
		f.write(str(tensor.shape[i]) + ' ')

	cnt = 30
	for i in tensor.flat:
		s = str(i) + ' '
		cnt += len(s)
		f.write(s)
		if cnt > 120:
			cnt = 0
			f.write('\\\n')

	f.write('";')
	f.write('\n\n')


if args.checkpoint is not None:
	if not args.dump:
		print("Restoring from checkpoint...")
	saver.restore(sess, args.checkpoint)

if args.dump:
	f = open("net.h", 'w')
	f.write("#include <string>\n\n")
	f.write("// Trained using checkpoint: " + args.checkpoint + "\n\n")
	write_tensor(f, net.W_conv1)
	write_tensor(f, net.b_conv1)

	write_tensor(f, net.W_conv2)
	write_tensor(f, net.b_conv2)

	write_tensor(f, net.W_conv3)
	write_tensor(f, net.b_conv3)

	write_tensor(f, net.W_conv4)
	write_tensor(f, net.b_conv4)

if args.visualize:
	# Dummy (todo: create empty)
	game = goinput.next_game()

	while True:
		command = input()

		if command == "visualize":
			words = input().strip().split(' ')

			fs = [float(x) for x in words]
			ps = [fs[x + y*19] for y in range(0,19) for x in range(0,19)]
			print(game.probabilities(ps, -1, -1, -1))


if args.simulate:
	game = goinput.next_game()

	while True:
		inp, label, move = goinput.input_from_game(game)
		if inp == None:
			game = goinput.next_game()
			continue

		scores = net.scores(sess, [inp], [label])[0]
		print(scores)

		maxcoord = scores.argmax()
		bestx = maxcoord % goutil.Board.SIZE
		besty = maxcoord // goutil.Board.SIZE
		print(game.path)
		print("Best score: " + str(scores.max()))
		print(game.probabilities(scores, move[0], move[1], move[2]))

		h_conv1 = sess.run(net.h_conv1, feed_dict={net.x: [inp], net.y_: [label]})
		test = scores[:]
		for i in range(0,19):
			for j in range(0,19):
				for k in range(0,32):
					if k == 0:
						test[i + j*19] = h_conv1[0][i][j][k]



		print(game.probabilities(test, -1, -1, -1))
		time.sleep(5.5)

if args.train:
	print("Training...")
	batch_size = 200
	for i in range(50000):
		batch_xs, batch_ys = goinput.next_batch(batch_size)

		net.train(sess, batch_xs, batch_ys)

		if i % 20 == 0:
			loss = net.loss(sess, batch_xs, batch_ys)
			print("Step: {0} Epoch: {1:.3f} Loss: {2:.2f}".format(i, goinput.epoch(), loss))

		if i % save_every == 0 and i > 0:
			print("Saving checkpoint...")
			print(saver.save(sess, "checkpoints/checkpoint", global_step=i))


# eval_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
#print("Final accuracy " + str(eval_accuracy))
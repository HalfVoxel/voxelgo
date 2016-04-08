import tensorflow as tf
import goinput


size = 19
pixels = size*size
input_size = pixels
input_channels = 3
output_size = pixels

x = tf.placeholder(tf.float32, [None, input_size*input_channels])
W = tf.Variable(tf.zeros([input_size*input_channels, input_size]))
b = tf.Variable(tf.zeros([output_size]))

# y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, output_size])

def weigth_var(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_var(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')



W_conv1 = weigth_var([5, 5, input_channels, 32])
b_conv1 = bias_var([32])
x_image = tf.reshape(x, [-1, size, size, input_channels])

# Shape [?, size, size, 32]
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

# W_conv2 = weigth_var([5, 5, 32, 64])
# b_conv2 = bias_var([64])

# h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# h_pool2 = max_pool_2x2(h_conv2)

# W_fc1 = weigth_var([7 * 7 * 64, 1024])
# b_fc1 = bias_var([1024])

# h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
# h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# keep_prob = tf.placeholder(tf.float32)
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Flatten
h_conv1_flat = tf.reshape(h_conv1, [-1, size*size*32])

# Fully connected layer #1
W_fc1 = weigth_var([size*size*32, size*size*8])
b_fc1 = bias_var([size*size*8])

h_fc1 = tf.nn.softmax(tf.matmul(h_conv1_flat, W_fc1) + b_fc1)

# Fully connected layer #2
W_fc2 = weigth_var([size*size*8, output_size])
b_fc2 = bias_var([output_size])

y = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(200):
	batch_size = 200
	batch_xs, batch_ys = goinput.next_batch(batch_size)
	if i % 10 == 0:
		#eval_accuracy = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
		loss = sess.run(cross_entropy, feed_dict={x: batch_xs, y_: batch_ys}) / batch_size
		print("Step " + str(i) + " loss " + str(loss))

	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# eval_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
#print("Final accuracy " + str(eval_accuracy))
import tensorflow as tf
import goinput
import time

size = 19
pixels = size*size
input_size = pixels
input_channels = 3
output_size = pixels

x = tf.placeholder(tf.float32, [None, input_size*input_channels], name="x")
y_ = tf.placeholder(tf.float32, [None, output_size], name="y_")

def weigth_var(shape, name):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial, name=name)

def bias_var(shape, name):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial, name=name)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')



W_conv1 = weigth_var([5, 5, input_channels, 64], "W_conv1")
b_conv1 = bias_var([64], "b_conv1")
x_image = tf.reshape(x, [-1, size, size, input_channels])

# Shape [?, size, size, 32]
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)


W_conv2 = weigth_var([5, 5, 64, 32], "W_conv2")
b_conv2 = bias_var([32], "b_conv2")
h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)

W_conv3 = weigth_var([5, 5, 32, 8], "W_conv3")
b_conv3 = bias_var([8], "b_conv3")
h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3)

# h_pool2 = max_pool_2x2(h_conv2)

# W_fc1 = weigth_var([7 * 7 * 64, 1024])
# b_fc1 = bias_var([1024])

# h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
# h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# keep_prob = tf.placeholder(tf.float32)
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_conv4 = weigth_var([5, 5, 8, 1], "W_conv4")
b_conv4 = bias_var([1], "b_conv4")
h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)

h_conv4_flat = tf.reshape(h_conv4, [-1, size*size])
y = tf.nn.softmax(h_conv4_flat)
# Flatten
# h_conv3_flat = tf.reshape(h_conv3, [-1, size*size*4])

# Fully connected layer #1
# W_fc1 = weigth_var([size*size*4, size*size])
# b_fc1 = bias_var([size*size])

# y = tf.nn.softmax(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)



cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.AdamOptimizer(0.002).minimize(cross_entropy)
#train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()
save_every = 100

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

if input("Restore from checkpoint? [y/N]").strip() == "y":
	saver.restore(sess, "checkpoints/checkpoint-300")

for i in range(50000):
	batch_size = 200
	t1 = time.clock()
	try:
		batch_xs, batch_ys = goinput.next_batch(batch_size)
	except:
		print("Generating batch failed. Skipping...")
		continue
	t2 = time.clock()

	if i % 20 == 0:
		#eval_accuracy = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
		t5 = time.clock()
		loss = sess.run(cross_entropy, feed_dict={x: batch_xs, y_: batch_ys}) / batch_size
		print("Step: {0} Epoch: {1:.3f} Loss: {2:.2f}".format(i, goinput.epoch(), loss))
		t6 = time.clock()

	if i % save_every == 0:
		print("Saving checkpoint...")
		print(saver.save(sess, "checkpoints/checkpoint", global_step=i))

	t3 = time.clock()
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
	t4 = time.clock()


# eval_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
#print("Final accuracy " + str(eval_accuracy))
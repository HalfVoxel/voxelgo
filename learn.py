import tensorflow as tf
import time
import argparse
import sys
import math

import goinput
import goutil
import gopolicynet
import govaluenet

parser = argparse.ArgumentParser(description="Go Neural Network")
parser.add_argument('--checkpoint', dest="checkpoint", default=None, help="path to checkpoint file")
parser.add_argument('--simulate', dest="simulate", action='store_true', help="simulate random games")
parser.add_argument('--policynet', dest="policynet", action='store_true', help="use the policy network")
parser.add_argument('--valuenet', dest="valuenet", action='store_true', help="use the value network")
parser.add_argument('--train', dest="train", action='store_true', help="train the net")

parser.add_argument('--dump', dest="dump", action='store_true', help="dump training parameters to output")
parser.add_argument('--visualize', dest="visualize", action='store_true', help="visualize probabilities from stdin")

args = parser.parse_args()

if args.valuenet:
	net = govaluenet.Net()
else:
	net = gopolicynet.Net()

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
	if args.valuenet:
		exit(1)

	# Dummy (todo: create empty)
	game = goinput.next_game()
	last_input = None

	while True:
		command = input()

		if command == "print":
			print("Printed: ")
			print(input())

		if command == "input":
			print("INPUT")
			words = input().strip().split(' ')
			fs = [float(x) for x in words]
			last_input = fs
			scores = net.scores(sess, [fs])[0]
			print(game.probabilities(scores, -1, -1, -1))

		if command == "board":
			words = input().strip().split(' ')

			fs = [int(x) for x in words]
			stones = [fs[x + y*19] for y in range(0,19) for x in range(0,19)]
			game = goutil.Board.from_stones(stones)
			assert(game.is_blacks_turn())
			inp = goinput.input_from_game(game)

			for i in range(0, len(last_input)):
				if abs(last_input[i] - inp[i]) > 0.1:
					print(" " + str(last_input[i]) + " " + str(inp[i]))
					exit(1)

			#print(inp)
			print(stones)
			scores = net.scores(sess, [inp])[0]
			print(game.probabilities(scores, -1, -1, -1))

		if command == "visualize":
			words = input().strip().split(' ')

			fs = [float(x) for x in words]
			ps = [fs[x + y*19] for y in range(0,19) for x in range(0,19)]
			print(game.probabilities(ps, -1, -1, -1))


if args.simulate:
	if args.valuenet:
		exit(1)

	game = goinput.next_game()

	while True:
		inp = goinput.input_from_game(game)
		# Must run after input_from_game
		move = game.next()
		if move == None:
			game = goinput.next_game()
			continue

		label = goinput.label_from_game(move)

		scores = net.scores(sess, [inp])[0]
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
	labeltype = "winner" if args.valuenet else "move"
	print("Training...")
	batch_size = 200
	for i in range(5000000):
		batch_xs, batch_ys = goinput.next_batch(batch_size, labeltype)
		net.train(sess, batch_xs, batch_ys)

		if i % 20 == 0:
			loss = net.loss(sess, batch_xs, batch_ys)
			print("Step: {0} Epoch: {1:.3f} Loss: {2:.2f}".format(i, goinput.epoch(), loss))

		if i % save_every == 0 and i > 0:
			print("Saving checkpoint...")
			print(saver.save(sess, "checkpoints/checkpoint", global_step=i))


# eval_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
#print("Final accuracy " + str(eval_accuracy))
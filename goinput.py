import goutil
import numpy
import tensorflow as tf
import random

games = goutil.load("train_alt")

current_games = []

def epoch():
	return games.epoch()

def next_game():
	return games.next(1)[0]

def input_from_game(game):
	white = [(1 if x == 1 else 0) for x in game.stones]
	black = [(1 if x == -1 else 0) for x in game.stones]
	freedoms = game.all_freedoms()

	move = game.next()

	if move == None:
		# End of game
		return None, None, None

	# Make sure that it is always black that makes the move
	if move[2] == 1:
		tmp = black
		black = white
		white = tmp

	# One-hot, flattened label
	label = [0]*goutil.Board.SIZE*goutil.Board.SIZE
	label[move[0] + move[1]*goutil.Board.SIZE] = 1

	inp = []
	for y in range(0, goutil.Board.SIZE):
		for x in range(0, goutil.Board.SIZE):
			i = x + y*goutil.Board.SIZE
			inp.append(black[i])
			inp.append(white[i])
			inp.append(freedoms[i])
			inp.append(x/(goutil.Board.SIZE-1) - 0.5)
			inp.append(y/(goutil.Board.SIZE-1) - 0.5)

	return inp, label, move

def next_batch(n):
	result = []
	result_labels = []

	while len(result) < n:
		try:
			while len(current_games) < n:
				current_games.append(next_game())

			# Pick random game
			game = random.choice(current_games)

			inp, label, move = input_from_game(game)

			if inp == None:
				current_games.remove(game)
				continue

			result.append(inp)
			result_labels.append(label)
		except Exception as e:
			print(e)
			print("Failed to generate input, skipping...")
			continue
	
	return result, result_labels
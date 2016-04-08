import goutil
import numpy
import tensorflow as tf
import random

games = goutil.load("train")

current_games = []

def next_batch(n):
	result = []
	result_labels = []

	while len(result) < n:
		while len(current_games) < n:
			current_games.append(games.next(1)[0])

		# Pick random game
		game = random.choice(current_games)

		white = [(1 if x == 1 else 0) for x in game.stones]
		black = [(1 if x == -1 else 0) for x in game.stones]
		freedoms = game.all_freedoms()

		dup = game.copy()
		try:
			move = game.next()
		except:
			print(str(dup))
			raise

		if move == None:
			# End of game
			current_games.remove(game)
			continue

		# Make sure that it is always black that makes the move
		if move[2] == 1:
			tmp = black
			black = white
			white = tmp

		# One-hot, flattened label
		label = [0]*goutil.Board.SIZE*goutil.Board.SIZE
		label[move[0] + move[1]*goutil.Board.SIZE] = 1

		inp = []
		for i in range(0, len(black)):
			inp.append(black[i])
			inp.append(white[i])
			inp.append(freedoms[i])

		result.append(inp)
		result_labels.append(label)
	
	return result, result_labels
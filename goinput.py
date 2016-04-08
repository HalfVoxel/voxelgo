import goutil
import numpy
import tensorflow as tf

games = goutil.load()

current_games = []

def next_batch(n):
	while len(current_games) < n:
		current_games.append(games.next(1))

	
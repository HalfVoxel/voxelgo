import sgf
import os
import time
import random

class Board:
	SIZE = 19
	dx = [1, 0, -1, 0]
	dy = [0, 1, 0, -1]

	def __init__(self, path, game_tree):
		self.path = path
		self.game_tree = game_tree
		if game_tree is not None:
			self.node = game_tree.root
		else:
			self.node = None
		self.stones = [0]*Board.SIZE*Board.SIZE

	def next(self):
		self.node = self.node.next
		if self.node == None:
			return None

		color = 0
		if 'W' in self.node.properties:
			pos = self.node.properties['W'][0]
			color = 1
		else:
			pos = self.node.properties['B'][0]
			color = -1

		if len(pos) != 2:
			print("Invalid move position: " + str(pos) + "\n" + self.path)
			return None

		x = ord(pos[0]) - ord('a')
		y = ord(pos[1]) - ord('a')
		if not self.place(x, y, color):
			print(self.highlight(x, y, color))
			print("Invalid Move at " + str(x) + " " + str(y) + "\n" + self.path)
			return None

		return (x, y, color)

	def copy(self):
		other = Board(self.path, self.game_tree)
		other.node = self.node
		# Copy stones
		other.stones = self.stones[:]
		return other

	def place(self, x, y, color):
		# Out of range
		if self.get(x,y) == -2:
			return False

		# Already a stone there
		if self.stones[x + y*Board.SIZE] != 0:
			return False
		else:
			self.stones[x + y*Board.SIZE] = color

			# Remove opponent stones first
			for i in range(0, 4):
				nx = x + Board.dx[i]
				ny = y + Board.dy[i]
				if self.get(nx, ny) == -color:
					self.remove_if_taken(nx, ny)

			# Then remove our stones
			for i in range(0, 4):
				nx = x + Board.dx[i]
				ny = y + Board.dy[i]
				if self.get(nx, ny) == color:
					self.remove_if_taken(nx, ny)

			# Remove the placed stone if it is taken
			self.remove_if_taken(x, y)

			# Actually an invalid move
			if self.get(x, y) != color:
				return False

			return True

	def get(self, x, y):
		if x < 0 or y < 0 or x >= Board.SIZE or y >= Board.SIZE:
			return -2

		return self.stones[x + y*Board.SIZE]

	def freedoms(self, x, y, seen_buffer):
		color = self.get(x,y)

		if color == 0 or color == -2:
			return 0

		seen_buffer.add((x,y))
		seen_freedoms = set()
		stack = [(x,y)]
		free = 0
		while len(stack) > 0:
			p = stack.pop()

			for i in range(0,4):
				po = (p[0]+Board.dx[i], p[1] + Board.dy[i])
				other = self.get(po[0], po[1])
				if other == 0:
					if po not in seen_freedoms:
						seen_freedoms.add(po)
						free += 1
				elif other == color:
					if po not in seen_buffer:
						seen_buffer.add(po)
						stack.append(po)

		return free

	def remove_if_taken(self, x, y):
		seen = set()
		fr = self.freedoms(x, y, seen)
		if fr == 0:
			# No free edges
			for p in seen:
				self.stones[p[0] + p[1]*Board.SIZE] = 0


	def all_freedoms(self):
		numfree = [-1]*Board.SIZE*Board.SIZE

		for y in range(0, Board.SIZE):
			for x in range(0, Board.SIZE):
				if numfree[x + y*Board.SIZE] == -1:
					if self.get(x,y) == 0:
						numfree[x + y*Board.SIZE] = 0
					else:
						seen = set()
						fr = self.freedoms(x, y, seen)
						for p in seen:
							numfree[p[0] + p[1]*Board.SIZE] = fr
		return numfree

	def __str__(self):
		return self.highlight(-1, -1, 0)

	def highlight(self, hx, hy, hcol):
		return self.probabilities(None, hx, hy, hcol)

	def probabilities(self, probs, hx, hy, hcol):
		s = ""
		for x in range(0, Board.SIZE):
			s += ' ' + str(x)
		s += '\n'
		s += ' ' + '--' * Board.SIZE + '\n'

		# Used with index 1 and -1
		#          White        Black
		COL = ['', '\033[94m', '\033[91m']
		HIGHLIGHT = ['', '\033[5m\033[94m', '\033[5m\033[91m']
		RESET = '\033[0m'

		if probs is not None:
			try:
				probmax = probs.max()
			except:
				# Eh, maybe it was a regular list
				probmax = max(probs)


		# ◯
		# ◉
		for y in range(0, Board.SIZE):
			s += '|'
			for x in range(0, Board.SIZE):
				col = self.stones[x + y*Board.SIZE]
				if probs is not None:
					p = probs[x + y*Board.SIZE] / probmax
					s += '\033[48;2;0;' + str(int(p * 255)) + ';0m'

				if x == hx and y == hy:
					if col != 0:
						s += HIGHLIGHT[hcol] + "◉ " + RESET
					else:
						s += HIGHLIGHT[hcol] + "◉ " + RESET
				else:
					if col != 0:
						s += COL[col] + "◯ " + RESET
					else:
						s += "  " + RESET

			s += ' |\n'

		s += ' ' + '--' * Board.SIZE
		return s


class Collection:
	def __init__(self, estimated_count, games):
		self.games = games
		self.count = 0
		self.estimated_count = estimated_count

	def epoch(self):
		return self.count / self.estimated_count

	def next(self, n):
		self.count += n
		return [next(self.games) for i in range(0, n)]

def iterate_valid_games_loop(paths):
	for game in iterate_games_loop(paths):
		gt = game.game_tree
		if 'SZ' in gt.root.properties:
			size = gt.root.properties['SZ']
			if size != ['19']:
				# print("Invalid board size (" + str(size) + ")")
				continue
		else:
			# TODO: Guess board size?
			# print("No board size specified")
			continue

		yield game


def all_paths(directory):
	all_paths = []
	for (path, dirs, files) in os.walk(directory):
		for file in files:
			if file.endswith('.sgf'):
				all_paths.append(os.path.join(path, file))

	random.shuffle(all_paths)
	return all_paths

def iterate_games_loop(paths):
	while(True):
		for filepath in paths:
			try:
				game = sgf.parse(open(filepath).read())
				gt = game.children[0]
				yield Board(filepath, gt)
			except Exception as e:
				print("Failed to parse " + filepath)
				print(e)


def load(directory):
	paths = all_paths(directory)
	return Collection(len(paths), iterate_valid_games_loop(paths))

if __name__ == "__main__":
	c = load("train_alt")
	for game in c.next(5):

		while True:
			move = game.next()
			if move is None:
				break

			print(game.highlight(move[0], move[1], move[2]))
			time.sleep(0.2)


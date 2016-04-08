import sgf
import os
import time
import random

class Board:
	SIZE = 19
	dx = [1, 0, -1, 0]
	dy = [0, 1, 0, -1]

	def __init__(self, game_tree):
		self.game_tree = game_tree
		self.node = game_tree.root
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

		x = ord(pos[0]) - ord('a')
		y = ord(pos[1]) - ord('a')
		if not self.place(x, y, color):
			print(self.highlight(x, y, color))
			print("Invalid Move at " + str(x) + " " + str(y))
			return None

		return (x, y, color)

	def copy(self):
		other = Board(self.game_tree)
		other.node = self.node
		# Copy stones
		other.stones = self.stones[:]
		return other

	def valid_move(self, x, y, color):
		if self.stones[x + y*Board.SIZE] != 0:
			return False

		for i in range(0, 4):
			v = get(x+Board.dx[i], y + Board.dy[i])
			if v == color or v == 0:
				return True

		return False

	def place(self, x, y, color):

		if self.stones[x + y*Board.SIZE] != 0:
			return False
		else:
			self.stones[x + y*Board.SIZE] = color

			# Remove opponent stones first
			for i in range(0, 4):
				if self.get(x + Board.dx[i], y + Board.dy[i]) == -color:
					self.remove_if_taken(x + Board.dx[i], y + Board.dy[i])

			# Then remove our stones
			for i in range(0, 4):
				if self.get(x + Board.dx[i], y + Board.dy[i]) == color:
					self.remove_if_taken(x + Board.dx[i], y + Board.dy[i])

			# Remove the placed stone
			self.remove_if_taken(x, y)

			# Actually an invalid move
			if self.stones[x + y*Board.SIZE] != color:
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
		stack = [(x,y)]
		free = 0
		while len(stack) > 0:
			p = stack.pop()

			for i in range(0,4):
				other = self.get(p[0]+Board.dx[i], p[1] + Board.dy[i])
				if other == 0:
					free += 1
				elif other == color:
					po = (p[0]+Board.dx[i], p[1] + Board.dy[i])
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
		frs = [-1]*Board.SIZE*Board.SIZE

		for y in range(0, Board.SIZE):
			for x in range(0, Board.SIZE):
				if frs[x + y*Board.SIZE] == -1:
					if self.get(x,y) == 0:
						frs[x + y*Board.SIZE] = 0
					else:
						seen = set()
						fr = self.freedoms(x, y, seen)
						for p in seen:
							frs[p[0] + p[1]*Board.SIZE] = fr
		return frs

	def __str__(self):
		return self.highlight(-1, -1, 0)

	def highlight(self, hx, hy, hcol):
		s = ""
		s += ' ' + '--' * Board.SIZE + '\n'

		# Used with index 1 and -1
		#          White        Black
		COL = ['', '\033[94m', '\033[91m']
		HIGHLIGHT = ['', '\033[5m\033[94m', '\033[5m\033[91m']
		RESET = '\033[0m'

		# ◯
		# ◉
		for y in range(0, Board.SIZE):
			s += '|'
			for x in range(0, Board.SIZE):
				col = self.stones[x + y*Board.SIZE]

				if x == hx and y == hy:
					if col != 0:
						s += HIGHLIGHT[hcol] + " ◉" + RESET
					else:
						s += HIGHLIGHT[hcol] + " ◉" + RESET
				else:
					if col != 0:
						s += COL[col] + " ◉" + RESET
					else:
						s += "  "

			s += ' |\n'

		s += ' ' + '--' * Board.SIZE
		return s


class Collection:
	def __init__(self, games):
		self.games = games

	def next(self, n):
		result = []
		for i in range(0, n):
			result.append(next(self.games))
		return result

def iterate_valid_games_loop(directory):
	for game in iterate_games_loop(directory):
		gt = game.children[0]
		if 'SZ' in gt.root.properties:
			size = gt.root.properties['SZ']
			if size != ['19']:
				print("Invalid board size (" + str(size) + ")")
				continue
		else:
			print("No board size specified")
			continue

		yield Board(gt)



def iterate_games_loop(directory):
	while(True):
		all_paths = []
		for (path, dirs, files) in os.walk(directory):
			for file in files:
				if file.endswith('.sgf'):
					all_paths.append(os.path.join(path, file))

		random.shuffle(all_paths)
		for filepath in all_paths:
			try:
				yield sgf.parse(open(filepath).read())
			except Exception as e:
				print("Failed to parse " + filepath)
				print(e)


def load(directory):
	return Collection(iterate_valid_games_loop(directory))

if __name__ == "__main__":
	c = load("train")
	for game in c.next(5):

		while True:
			move = game.next()
			if move is None:
				break

			print(game.highlight(move[0], move[1], move[2]))
			time.sleep(0.2)


import goutil
import sgf

path = input("File: ")
board = goutil.Board(path, sgf.parse(open(path).read()).children[0])

while True:
	try:
		m = input("Move: ")
		if m == "n":
			nb = board.copy()
			move = nb.next()
			if move is not None:
				board = nb
				print(board.highlight(move[0], move[1], move[2]))
		else:
			ms = m.split(' ')
			x = int(ms[0])
			y = int(ms[1])
			c = int(ms[2])
			if c == 0:
				c = -1
			nb = board.copy()
			if not nb.place(x, y, c):
				print("Invalid move")
			else:
				board = nb
				print(board.highlight(x, y, c))
	except Exception as e:
		print(e)

# train_alt/Shusai/Shusai-723.sgf
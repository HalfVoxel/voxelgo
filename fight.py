import goinput
import goutil
import sys
from subprocess import Popen, PIPE, DEVNULL
import time
import random

total_scores = [0,0]

programs = ["a.out", "b.out"]

while True:
	players = []
	prog1 = random.choice([0,1])
	prog2 = 1 - prog1
	players.append(Popen(programs[prog1], stdin=PIPE, stdout=PIPE, stderr=DEVNULL))

	# Sleep a small amount of time to avoid starting the programs at the same time
	# which could give them the same seed in their RNGs (I think...)
	time.sleep(0.1)
	players.append(Popen(programs[prog2], stdin=PIPE, stdout=PIPE, stderr=DEVNULL))

	for i in range(0,2):
		players[i].stdin.write(('settings your_botid ' + str(i+1) + '\n').encode('utf-8'))

	game = goutil.Board("/fight/", None)

	player = 0
	passes = 0
	turn = 0
	while True:
		turn += 1
		if turn > 700:
			print("Turn limit reached")
			break

		field = ','.join([('1' if x == -1 else ('2' if x == 1 else '0')) for x in game.stones])
		players[player].stdin.write(('update game field ' + field + '\n\n').encode('utf-8'))
		players[player].stdin.write(b'action move 1000\n\n')
		players[player].stdin.flush()

		cmd = players[player].stdout.readline().decode('utf-8').strip().split(' ')
		if cmd[0] == 'place_move':
			passes = 0
			x = int(cmd[1])
			y = int(cmd[2])
			col = -1 if player == 0 else 1
			game.place(x, y, col)
			#print(game.highlight(x, y, col))
		else:
			assert(cmd[0] == "pass")
			#print("Pass")

			passes += 1
			if passes >= 2:
				break

		player = 1 if player == 0 else 0
		sys.stdout.flush()

	if turn > 700:
		continue

	print(game.scores())
	print(game.highlight(-1,-1,-1))
	winner = game.current_leader()
	winner_program = prog1 if winner == -1 else prog2
	print(str(winner) + " : " + str(winner_program))
	total_scores[winner_program] += 1
	print(total_scores)
	print()

	for player in players:
		player.terminate()

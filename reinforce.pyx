import goutil
from goutil import SIZE, Board
import math
import goinput
import random
import time

replay_memory = []
late_game_replay_memory = []
validation_memory = []

class ReinforcedHistory:
    def __init__(self):
        self.history = []
        self.moves = []
        self.decay_factor = 0.03

    def new_move(self, x, y):
        self.moves.append((x, y))
        self.history.append(0)

    def reinforce(self, delta):
        for i in range(0, len(self.history)):
            # Moves since this move was done
            delta_moves = len(self.history) - i - 1
            self.history[i] += delta * math.exp(-self.decay_factor*delta_moves)


class ReinforcedGame:
    def __init__(self):
        self.game = Board("/reinforced/", None)
        self.last_scores = [0, 0]
        # self.histories = [ReinforcedHistory(), ReinforcedHistory()]
        self.passes = 0
        self.moves = []

    def is_over(self):
        return self.passes >= 2

    def pass_turn(self, color):
        if self.is_over():
            raise("Game is already over")

        # histories[0 if color == -1 else 1].new_move(19, 19)
        self.game.pass_turn(color)
        self.passes += 1

    def end_game(self):
        print("Game Completed")
        self.passes = 2
        self.construct_replay_memory()

    def construct_replay_memory(self):
        global replay_memory
        global late_game_replay_memory
        sim = Board("/reinforced/", None)
        for i in range(0, len(self.moves)):
            move, valid = self.moves[i]

            origstones = sim.stones[:]
            terminal = (i == len(self.moves) - 1)

            if valid:
                    sim.make_move_or_pass(move[0], move[1], move[2])
            else:
                sim.make_move_or_pass(19, 19, move[2])

            # if not self.game.place(move[0], move[1], move[2]):
            #   # Invalid move, fill the board with the opponent's color
            #   self.game.stones = [-color] * SIZE * SIZE
            #   terminal = True
            #   break
            if terminal:
                scores = sim.scores_without_komi()
                if move[2] == -1:
                    # Black score - white score
                    # reward = 1 if scores[0] > scores[1] else -1
                    reward = scores[0] - scores[1]
                    # print(sim.highlight(-1, -1, -1))
                    # print("Black wins by " + str(reward))
                else:
                    assert(move[2] == 1)
                    # White score - black score
                    # reward = 1 if scores[1] > scores[0] else -1
                    reward = scores[1] - scores[0]
                    # print(sim.highlight(-1, -1, -1))
                    # print("White wins by " + str(reward))

            else:
                # Check if it was a pass (invalid move)
                if not valid:
                    # Invalid moves are bad!
                    # Say equivalent to losing N stones
                    reward = -10
                else:
                    reward = 0

            replay = (origstones, move, sim.stones[:], reward, terminal)
            # replay_memory.append(replay)

            # replay_mirror = ([-x for x in origstones], move, [-x for x in sim.stones], -reward, terminal)
            # replay_memory.append(replay_mirror)

            if len(self.moves) - i <= 4:
                if random.randint(0, 5) == 0:
                    validation_memory.append(replay)
                else:
                    late_game_replay_memory.append(replay)

    def any_reasonable_move(self, game):
        valid = game.all_valid_moves()
        for x in valid:
            if x == 1:
                return True

        return False

    def place(self, x, y, color):
        if self.is_over():
            raise("Game is already over")

        self.passes = 0
        # new_scores = game.scores()
        move = (x, y, color)

        pass_move = x == 19 and y == 19
        assert(self.game.is_blacks_turn == (color == -1))
        dup = self.game.copy()
        valid = True
        try:
            dup.make_move_or_pass(x, y, color)
        except Exception:
            valid = False
            # Invalid move, treat as pass
            # print("Invalid move... passing\n" + str(e))

        if valid and not pass_move:
            self.game.make_move_or_pass(x, y, color)
        else:
            # No valid moves for this player
            # (there may still be valid moves for the other player, but we will ignore those)
            if pass_move:
                self.end_game()
                return

            # Games are pretty unlikely to end before turn 200
            if len(self.moves) >= 200:
                # Check if there no reasonable moves
                # and if so, end the game
                #valid = self.game.all_valid_moves()
                #print(self.game.probabilities(valid, -1, -1, -1))
                if not self.any_reasonable_move(self.game):
                    self.end_game()
                    return

            self.game.make_move_or_pass(19, 19, color)

        self.moves.append((move, valid))

        # Turn limit
        if len(self.moves) >= 500 and random.randint(0, 1) == 1:
            self.end_game()

        # Invalid move, surrender

        # histories[0 if color == -1 else 1].new_move(x, y)

        # for i in range(0,2):
        #   histories[i].reinforce(new_scores[i] - last_scores[i])

    def get_input(self):
        return goinput.input_from_game(self.game)

current_games = []


def next_reinforced_game_batch(n):
    global current_games
    # Remove completed games
    current_games = [game for game in current_games if not game.is_over()]

    # Keep a pool of batch size games
    while len(current_games) < n:
        current_games.append(ReinforcedGame())

    return current_games[0:n]


def next_replay_batch(n, validation = False):
    mem = validation_memory if validation else late_game_replay_memory
    if len(mem) < n:
        return [], [], [], []

    t1 = 0
    t2 = 0
    t3 = 0
    if validation:
        sample = mem[0:n]
    else:
        sample = random.sample(mem, n)
    inputs1 = []
    inputs2 = []
    actions = []
    rewards = []

    for replay in sample:
        s1 = time.time()
        origstones, move, stones, reward, terminal = replay
        orig_game = Board.from_stones(origstones)

        orig_game.is_blacks_turn = move[2] == -1
        next_game = Board.from_stones(stones)
        next_game.is_blacks_turn = not orig_game.is_blacks_turn
        s2 = time.time()
        t1 += s2 - s1

        inp1 = goinput.input_from_game(orig_game)
        inp2 = goinput.input_from_game(next_game)
        s3 = time.time()
        t2 += s3 - s2

        action = [0]*SIZE*SIZE
        pass_move = move[0] >= SIZE or move[1] >= SIZE
        assert(not pass_move)
        if not pass_move:
            action[move[1]*SIZE + move[0]] = 1

        inputs1.append(inp1)
        inputs2.append(inp2)
        actions.append(action)
        rewards.append((reward, terminal))

        s4 = time.time()
        t3 += s4 - s3

    # print(str(t1) + " " + str(t2) + " " + str(t3))

    return inputs1, inputs2, actions, rewards

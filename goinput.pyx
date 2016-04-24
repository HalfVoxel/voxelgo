import goutil
import random
import traceback
import time

games = goutil.load("train_kyu")

current_games = []


def epoch():
    return games.epoch()


def next_game():
    return games.next(1)[0]


def label_from_game(move):
    # One-hot, flattened label
    label = [0]*goutil.SIZE*goutil.SIZE
    label[move[0] + move[1]*goutil.SIZE] = 1
    return label


def winner_label_from_game(game, move):
    # -1 if it is not blacks move, otherwise 1
    # This makes sure that if the current player
    # wins the game, then the label will be -1
    # otherwise it will be 1
    blacks_move = move[2] == -1
    if game.game_tree is not None:
        props = game.game_tree.root.properties
        if "RE" in props:
            resultdata = props["RE"][0]
            if resultdata.startswith('W'):
                return [1 if not blacks_move else -1]
            elif resultdata.startswith('B'):
                return [1 if blacks_move else -1]

    return None


def input_from_game(game):
    cdef int x, y
    t1 = time.time()
    white = [(1 if x == 1 else 0) for x in game.stones]
    black = [(1 if x == -1 else 0) for x in game.stones]
    freedoms = game.all_freedoms()
    t2 = time.time()

    # Make sure that it is always black that makes the move
    if not game.is_blacks_turn:
        tmp = black
        black = white
        white = tmp

    inp = []
    for y in range(0, goutil.SIZE):
        for x in range(0, goutil.SIZE):
            i = x + y*goutil.SIZE
            inp.append(black[i])
            inp.append(white[i])
            inp.append(min(freedoms[i], 4))
            inp.append(x/(goutil.SIZE-1) - 0.5)
            inp.append(y/(goutil.SIZE-1) - 0.5)

    t3 = time.time()

    return inp


def next_batch(n, labeltype="move"):
    result = []
    result_labels = []

    while len(result) < n:
        try:
            # Keep a pool of 5*batch size games for variation
            while len(current_games) < 5*n:
                current_games.append(next_game())

            # Pick random game
            game = random.choice(current_games)

            # valid = True
            # while game.node.next is not None and game.node.next.next is not None:
            #   move = game.next()
            #   if move is None:
            #       current_games.remove(game)
            #       valid = False
            #       break

            # if not valid:
            #   continue

            # Keep a copy because the input_from_game
            # method requires the game state before the move was done
            # as well as the move.
            # This is slightly inefficient, but oh well
            dup = game.copy()

            move = game.next()

            if move is None:
                current_games.remove(game)
                continue

            inp = input_from_game(dup)

            if labeltype == "winner":
                label = winner_label_from_game(game, move)
                if label is None:
                    # No winner, boring!
                    current_games.remove(game)
                    continue
            else:
                assert(labeltype == "move")
                label = label_from_game(move)

            result.append(inp)
            result_labels.append(label)
        except Exception as e:
            traceback.print_exc(e)
            print("Failed to generate input, skipping...")
            continue

    return result, result_labels

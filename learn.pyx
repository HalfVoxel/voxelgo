import tensorflow as tf
import time
import argparse
import random
import sys

import goinput
import goutil
import gopolicynet
import govaluenet
import goqnet
import reinforce


def main(args):
    if args.valuenet:
        net = govaluenet.Net()
    elif args.qnet:
        net = goqnet.Net()
    else:
        net = gopolicynet.Net()

    saver = tf.train.Saver()
    save_every = 400

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    batch_size = args.batch_size

    def write_tensor(f, variable):
        name = variable.name.replace(":0", "") + "_data"
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
                stones = [fs[x + y * 19] for y in range(0, 19) for x in range(0, 19)]
                game = goutil.Board.from_stones(stones)
                assert(game.is_blacks_turn)
                inp = goinput.input_from_game(game)

                for i in range(0, len(last_input)):
                    if abs(last_input[i] - inp[i]) > 0.1:
                        print(" " + str(last_input[i]) + " " + str(inp[i]))
                        exit(1)

                # print(inp)
                print(stones)
                scores = net.scores(sess, [inp])[0]
                print(game.probabilities(scores, -1, -1, -1))

            if command == "visualize":
                words = input().strip().split(' ')

                fs = [float(x) for x in words]
                ps = [fs[x + y * 19] for y in range(0, 19) for x in range(0, 19)]
                print(game.probabilities(ps, -1, -1, -1))

    if args.simulate:
        if args.valuenet:
            exit(1)

        game = goinput.next_game()

        while True:
            inp = goinput.input_from_game(game)
            # Must run after input_from_game
            move = game.next()
            if move is None:
                game = goinput.next_game()
                continue

            label = goinput.label_from_game(move)

            scores = net.scores(sess, [inp])[0]
            print(scores)

            # maxcoord = scores.argmax()
            # bestx = maxcoord % goutil.SIZE
            # besty = maxcoord // goutil.SIZE
            print(game.path)
            print("Best score: " + str(scores.max()))
            print(game.probabilities(scores, move[0], move[1], move[2]))

            h_conv1 = sess.run(net.h_conv1, feed_dict={net.x: [inp], net.y_: [label]})
            test = scores[:]
            for i in range(0, 19):
                for j in range(0, 19):
                    for k in range(0, 32):
                        if k == 0:
                            test[i + j * 19] = h_conv1[0][i][j][k]

            print(game.probabilities(test, -1, -1, -1))

            time.sleep(5.5)

    
    if args.train and (args.policynet or args.valuenet):
        labeltype = "winner" if args.valuenet else "move"
        print("Training...")
        for i in range(5000000):
            print("Generating input...")
            batch_xs, batch_ys = goinput.next_batch(batch_size, labeltype)
            print("Training batch...")
            net.train(sess, batch_xs, batch_ys)

            if i % 20 == 0:
                loss = net.loss(sess, batch_xs, batch_ys)
                print("Step: {0} Epoch: {1:.3f} Loss: {2:.2f}".format(i, goinput.epoch(), loss))

            if i % save_every == 0 and i > 0:
                print("Saving checkpoint...")
                print(saver.save(sess, "checkpoints/checkpoint", global_step=i))

    def sample_random_move(game, color):
        for i in range(0, 5):
            x = random.randrange(0, 19)
            y = random.randrange(0, 19)
            if game.game.is_reasonable_move(x, y, color):
                return x, y

        valid = game.game.all_valid_moves()
        for x in range(0, 19):
            for y in range(0, 19):
                if valid[x + y*19] == 1:
                    return x, y
                #if game.game.is_reasonable_move(x, y, color):
                #    return x, y

        return None, None

    def generate_replays(batch_size, probability_to_make_random_move):
        # print("Generating replays")
        games = reinforce.next_reinforced_game_batch(batch_size)
        random_games = random.sample(games, int(len(games) * probability_to_make_random_move))
        games = [game for game in games if game not in random_games]

        timer1 = 0
        timer2 = 0
        timer3 = 0
        timer4 = 0

        for game in random_games:
            color = -1 if game.game.is_blacks_turn else 1
            t0 = time.time()
            x, y = sample_random_move(game, color)
            t1 = time.time()
            timer1 += t1 - t0

            if x is None:
                print("Could not find a valid random move")
                game.place(19, 19, color)
            else:
                game.place(x, y, color)

            t2 = time.time()
            timer2 += t2 - t1

        if len(games) > 0:
            t1 = time.time()
            inputs = [game.get_input() for game in games]
            moves = net.moves(sess, inputs)
            t2 = time.time()
            timer3 += t2 - t1

            for game, move in zip(games, moves):
                color = -1 if game.game.is_blacks_turn else 1
                game.place(move[0], move[1], color)

            t3 = time.time()
            timer4 += t3 - t2

        # print("Timers: {0:.2f}, {1:.2f}, {2:.2f}, {3:.2f}".format(timer1, timer2, timer3, timer4))

    def train_using_replays(batch_size, validation):
        inputs1, inputs2, actions, rewards = reinforce.next_replay_batch(batch_size, validation)

        # Batch could not be generated because there were too few replays
        if len(inputs1) == 0:
            # print("Too few replays for training: " + str(len(reinforce.replay_memory)))
            return

        # print("Training using replays")
        # Calculate Q values for the next state
        qs = net.max_qs(sess, inputs2)
        wrapped_qs = []
        # Calculate desired reward for this state
        gamma = 0.96
        for i in range(0, len(rewards)):
            reward, terminal = rewards[i]
            if terminal:
                q = reward
            else:
                # Note that the next state is seen from the perspective of the opponent
                # so the Q values must be negated to make them useful for the current player
                q = reward + gamma * (-qs[i])

            wrapped_qs.append(q)

        # If terminal
        if rewards[0][1]:
            player1 = (inputs1[0][x + 0] for x in range(0, 19 * 19 * 5, 5))
            player2 = (inputs1[0][x + 1] for x in range(0, 19 * 19 * 5, 5))
            # print(str(sum(player1)) + " " + str(sum(player2)) + " " + str(wrapped_qs[0][0]))
            # scores = net.scores(sess, inputs1)
            # print(scores)

        # Approximately assert that it is one-hot
        assert(sum(sum(x) for x in actions) == len(actions))

        q_taken = net.q_taken(sess, inputs1, actions)
        # print(list(zip(q_taken, wrapped_qs)))
        if validation:
            loss = net.eval_loss(sess, inputs1, actions, wrapped_qs)
            print("Validation Loss: " + str(loss))
        else:
            loss = net.train(sess, inputs1, actions, wrapped_qs)
            print("Training Loss: " + str(loss))

    if args.qnet:
        assert(not args.valuenet)
        assert(not args.policynet)

        for i in range(500000000):

            probability_to_make_random_move = min(max(1 - 0.0001 * (i - 500), 0.1), 1)
            if (i % 10) == 0:
                print("Batch #" + str(i) + " α=" + str(probability_to_make_random_move))
                train_using_replays(100, True)

            if i % save_every == 0 and i > 0:
                print("Saving checkpoint...")
                print(saver.save(sess, "checkpoints/checkpoint", global_step=i))

            generate_replays(batch_size, probability_to_make_random_move)
            train_using_replays(batch_size, False)

# eval_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
# print("Final accuracy " + str(eval_accuracy))


def run(stringargs):
    parser = argparse.ArgumentParser(description="Go Neural Network")
    parser.add_argument('--checkpoint', dest="checkpoint", default=None, help="path to checkpoint file")
    parser.add_argument('--simulate', dest="simulate", action='store_true', help="simulate random games")
    parser.add_argument('--policynet', dest="policynet", action='store_true', help="use the policy network")
    parser.add_argument('--valuenet', dest="valuenet", action='store_true', help="use the value network")
    parser.add_argument('--qnet', dest="qnet", action='store_true', help="use the Q network")
    parser.add_argument('--train', dest="train", action='store_true', help="train the net")

    parser.add_argument('--dump', dest="dump", action='store_true', help="dump training parameters to output")
    parser.add_argument('--visualize', dest="visualize", action='store_true', help="visualize probabilities from stdin")
    parser.add_argument('--batch-size', dest="batch_size", default=200, type=int, help="batch size")

    args = parser.parse_args(stringargs)
    main(args)

if __name__ == "__main__":
    run(sys.argv[1:])

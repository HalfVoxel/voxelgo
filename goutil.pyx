import sgf
import os
import time
import random

cdef inline int cget(int[19 * 19] stones, int x, int y):
    if x < 0 or y < 0 or x >= 19 or y >= 19:
        return -2

    return stones[x + y * 19]

cdef int[19 * 19] cached_scratch_board
# cached_scratch_board = None

cdef int[4] dx, dy
dx = [1, 0, -1, 0]
dy = [0, 1, 0, -1]

# cdef int SIZE
SIZE = 19


class Board:
    SIZE = 19
    # dx = [1, 0, -1, 0]
    # dy = [0, 1, 0, -1]

    @staticmethod
    def from_stones(stones):
        game = Board("/tmp/", None)
        game.set_stones(stones)
        return game

    @staticmethod
    def _moves_from_game_tree(game_tree):
        node = game_tree.root
        moves = []
        while node is not None:
            color = 0
            if 'W' in node.properties:
                pos = node.properties['W'][0]
                color = 1
            elif 'B' in node.properties:
                pos = node.properties['B'][0]
                color = -1
            else:
                # This is usually the last move of the game
                # which for some reason seems to be empty sometimes..
                node = node.next
                continue

            if len(pos) != 2:
                raise Exception("Invalid move position: " + str(pos))

            x = ord(pos[0]) - ord('a')
            y = ord(pos[1]) - ord('a')
            moves.append((x, y, color))
            node = node.next

        return moves

    def __init__(self, path, game_tree):
        self.path = path
        self.game_tree = game_tree
        self.is_blacks_turn = True
        if game_tree is not None:
            self.moves = Board._moves_from_game_tree(game_tree)
        else:
            self.moves = None
        self.stones = [0] * SIZE * SIZE
        self.turn = 0
        self.dead = None
        self.cached_components = [None, None]

    def set_stones(self, stones):
        assert(len(stones) == SIZE * SIZE)
        self.stones = stones
        self.is_blacks_turn = True
        self.turn = 0
        self.dead = None
        self.moves = None
        self.path = None
        self.game_tree = None
        self.dirty()

    def dirty(self):
        self.cached_components[0] = None
        self.cached_components[1] = None

    def next(self):
        if self.moves is None or self.turn >= len(self.moves):
            return None

        x, y, color = self.moves[self.turn]

        self.turn += 1
        self.dirty()

        # Apparently this is a pass
        if x == 19 and y == 19:
            self.pass_turn(color)
            return self.next()

        self.make_move(x, y, color)
        return (x, y, color)

    def get_dead(self):
        self.calculate_dead_states()
        return self.dead[self.turn]

    def connected_components(self, color):
        comps = [-1] * 19 * 19

        id = 1
        for x in range(0, 19):
            for y in range(0, 19):
                if comps[y * SIZE + x] == -1 and self.get(x, y) == color:
                    seen = []
                    info = self.freedoms(x, y, seen)

                    for p in seen:
                        comps[p[1] * SIZE + p[0]] = id

                    assert(info != 0)
                    id += 1

        return comps

    def is_our_eye(self, x, y, color):
        idx = 0 if color == -1 else 1
        if self.cached_components[idx] is not None:
            comps = self.cached_components[idx]
        else:
            comps = self.connected_components(color)
            self.cached_components[idx] = comps

        direct = 0
        different_groups = False
        groups_id = -1
        for i in range(0, 4):
            nx = x + dx[i]
            ny = y + dy[i]
            if self.get(nx, ny) == color:
                if groups_id == -1:
                    groups_id = comps[ny * SIZE + nx]
                elif comps[ny * SIZE + nx] != groups_id:
                    different_groups = True

                direct += 1
            elif self.get(nx, ny) == -2:
                direct += 1

        # Definitely qualifies as an eye
        return not different_groups and direct == 4

    def is_reasonable_move(self, x, y, color):
        return self.get(x, y) == 0 and not self.is_our_eye(x, y, color)

    def all_valid_moves(self):
        cdef int[19 * 19] valid, freedoms
        cdef int x, y, nx, ny, col, i

        freedoms = self.all_freedoms()
        valid = [0] * SIZE * SIZE
        for x in range(0, SIZE):
            for y in range(0, SIZE):
                # Loop through possible colors, either -1 or 1
                for col in range(-1, 2, 2):
                    if self.is_reasonable_move(x, y, col):
                        for i in range(0, 4):
                            nx = x + dx[i]
                            ny = y + dy[i]
                            frs = cget(freedoms, nx, ny)
                            stone = self.get(nx, ny)
                            if stone == 0 or (stone == -col and frs == 1) or (stone == col and frs > 1):
                                # Totally valid
                                # Either adjacent to an empty tile
                                # or adjacent to a group of enemy stones which have only a single freedom (this tile)
                                # or adjacent to a friendly group which has at least one other freedom
                                valid[x + y * 19] = 1

        return valid

    def calculate_dead_states(self):
        if self.dead is not None:
            return

        assert(self.turn == 0)

        self.dead = []
        stones = []
        dup = self.copy()
        for step in range(0, 100000):

            self.dead.append([0] * SIZE * SIZE)
            stones.append(dup.stones[:])

            move = dup.next()
            if move is None:
                break

            for i in range(0, len(dup.stones)):
                if stones[step][i] != 0 and dup.stones[i] == 0:
                    b = step

                    # Show as dead at most 20 steps back
                    MAX_HISTORY = 20

                    while b >= 0 and stones[b][i] != 0 and step - b < MAX_HISTORY:
                        self.dead[b][i] = 1
                        b -= 1

    def copy(self):
        other = Board(self.path, self.game_tree)
        # Copy stones
        other.stones = self.stones[:]
        other.turn = self.turn
        other.moves = self.moves
        other.is_blacks_turn = self.is_blacks_turn
        return other

    def make_move_or_pass(self, x, y, color):
        self.dirty()

        if x == 19 and y == 19:
            self.pass_turn(color)
        else:
            self.make_move(x, y, color)

    def pass_turn(self, color):
        if (color == -1) is not self.is_blacks_turn:
            raise Exception("The wrong player made the move")

        self.dirty()
        self.is_blacks_turn = not self.is_blacks_turn

    def make_move(self, x, y, color):
        if (color == -1) is not self.is_blacks_turn:
            raise Exception("The wrong player made the move")

        self.dirty()
        self.is_blacks_turn = not self.is_blacks_turn
        if not self._place(x, y, color):
            raise Exception("Invalid Move at " + str(x) + " " + str(y) + "\n" + self.path)

    def _place(self, int x, int y, int color):
        cdef int nx, ny, i

        assert(color == 1 or color == -1)
        self.dirty()

        # Out of range
        if self.get(x, y) == -2:
            return False

        # Already a stone there
        if self.stones[x + y * SIZE] != 0:
            return False
        else:
            self.stones[x + y * SIZE] = color

            # Remove opponent stones first
            for i in range(0, 4):
                nx = x + dx[i]
                ny = y + dy[i]
                if self.get(nx, ny) == -color:
                    self._remove_if_taken(nx, ny)

            # Then remove our stones
            for i in range(0, 4):
                nx = x + dx[i]
                ny = y + dy[i]
                if self.get(nx, ny) == color:
                    self._remove_if_taken(nx, ny)

            # Remove the placed stone if it is taken
            self._remove_if_taken(x, y)

            # Actually an invalid move
            if self.get(x, y) != color:
                return False

            return True

    def get(self, int x, int y):
        # Hot code: hard code size 19
        if x < 0 or y < 0 or x >= 19 or y >= 19:
            return -2

        return self.stones[x + y * 19]

    cached_id = 100000

    def freedoms(self, int x, int y, seen_buffer, int cutoff=10000):
        global cached_scratch_board
        cdef int id, nx, ny, free, color
        cdef int[19 * 19] stones
        stones = self.stones

        Board.cached_id += 1

        # Wrap
        if Board.cached_id > 100000:
            Board.cached_id = 1
            cached_scratch_board = [0] * 19 * 19

        id = Board.cached_id
        scratch = cached_scratch_board

        color = cget(stones, x, y)

        if color == 0 or color == -2:
            return 0

        seen_buffer.append((x, y))

        stack = [(x, y)]
        free = 0
        while len(stack) > 0:
            x, y = stack.pop()

            for i in range(0, 4):
                nx = x + dx[i]
                ny = y + dy[i]
                idx = ny * 19 + nx
                other = cget(stones, nx, ny)
                if other == 0:
                    # if po not in seen_freedoms:
                    if scratch[idx] != id:
                        # seen_freedoms.add(po)
                        scratch[idx] = id
                        free += 1
                        if free >= cutoff:
                            break
                elif other == color:
                    # if po not in seen_buffer:
                    if scratch[idx] != id:
                        po = (nx, ny)
                        seen_buffer.append(po)
                        scratch[idx] = id
                        stack.append(po)

        return free

    cached_remove_list = []

    def _remove_if_taken(self, int x, int y):
        seen = []
        fr = self.freedoms(x, y, seen, 1)
        if fr == 0:
            # No free edges
            for p in seen:
                self.stones[p[0] + p[1] * SIZE] = 0

    def all_freedoms(self):
        cdef int[19 * 19] stones
        stones = self.stones
        numfree = [-1] * SIZE * SIZE

        seen = []

        for y in range(0, SIZE):
            for x in range(0, SIZE):
                if numfree[x + y * SIZE] == -1:
                    if cget(stones, x, y) == 0:
                        numfree[x + y * SIZE] = 0
                    else:
                        seen.clear()
                        fr = self.freedoms(x, y, seen)
                        for p in seen:
                            numfree[p[0] + p[1] * SIZE] = fr
        return numfree

    def regions(self):
        regs = [0] * SIZE * SIZE

        for y in range(0, SIZE):
            for x in range(1, SIZE):
                col = self.stones[x + y * SIZE]

                reg = 0
                if col != 0:
                    reg = col
                elif y > 0 and regs[x + (y - 1) * SIZE] != 0:
                    reg = regs[x + (y - 1) * SIZE]
                elif regs[(x - 1) + y * SIZE] != 0:
                    reg = regs[(x - 1) + y * SIZE]

                regs[x + y * SIZE] = reg

        for y in range(SIZE - 1, -1, -1):
            for x in range(SIZE - 2, -1, -1):
                col = self.stones[x + y * SIZE]

                reg = 0
                if col != 0:
                    reg = col
                elif y < SIZE - 1 and regs[x + (y + 1) * SIZE] != 0:
                    reg = regs[x + (y + 1) * SIZE]
                elif regs[(x + 1) + y * SIZE] != 0:
                    reg = regs[(x + 1) + y * SIZE]

                regs[x + y * SIZE] = reg

        return regs

    def scores_without_komi(self):
        regs = self.regions()

        # Indexed using 0 (for black) and 1 (for white)
        score = [0, 0]
        for i in range(0, len(regs)):
            # 0 for black, 1 for white
            idx = max(regs[i], 0)
            score[idx] += 1

        return score

    def scores(self):
        score = self.scores_without_komi()

        komi = 7.5
        score[1] += komi
        return score

    def current_leader(self):
        sc = self.scores()
        return -1 if sc[0] > sc[1] else 1

    def __str__(self):
        return self.highlight(-1, -1, 0)

    def highlight(self, hx, hy, hcol):
        return self.probabilities(None, hx, hy, hcol)

    def probabilities(self, probs, hx, hy, hcol):
        s = ""
        # for x in range(0, SIZE):
        #   s += ' ' + str(x)
        # s += '\n'
        s += ' ' + '--' * SIZE + '\n'

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
        for y in range(0, SIZE):
            s += '|'
            for x in range(0, SIZE):
                col = self.stones[x + y * SIZE]
                if probs is not None:
                    p = probs[x + y * SIZE]
                    if probmax > 0:
                        p /= probmax
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

        s += ' ' + '--' * SIZE
        return s


class Collection:
    def __init__(self, estimated_count, games):
        self.games = games
        self.count = 0
        self.estimated_count = estimated_count

    def epoch(self):
        return self.count / self.estimated_count

    def next(self, n):
        print("Getting " + str(n) + " games...")
        self.count += n
        return [next(self.games) for i in range(0, n)]


def iterate_valid_games_loop(paths):
    for game in iterate_games_loop(paths):
        gt = game.game_tree
        if 'SZ' in gt.root.properties:
            size = gt.root.properties['SZ']
            if size != ['19']:
                print("Invalid board size (" + str(size) + ")")
                continue
        else:
            # TODO: Guess board size?
            # print("No board size specified")
            continue

        print("Yielding game")
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
                print("Parsing... " + filepath)
                game = sgf.parse(open(filepath).read())
                print("Parsed")
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

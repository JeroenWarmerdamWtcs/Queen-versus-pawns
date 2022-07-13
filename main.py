from itertools import combinations, product

# This is a Python script for solving the game "queen vs pawns":
# The game start with the white pawns and the black queen
# on the usual initial squares and those are the only pieces.
# White's goal is to promote a pawn. Capturing the queen is also a win.
# Black's goal is to capture all pawns before any promotes.
# The position is draw if there is no legal move, i.e. queen blocks last pawn.

# We might restrict queen moves, not to go to an attacked square.
#    There will always be legal moves left for the queen. Assume not, then:
#       - It cannot be on rank 1 as on rank 1 it can freely move
#       - The square below the queen must be under attack
#         by a pawn in a neighbouring file and two ranks lower
#       - Let us call the rank of the pawn r, one higher s and two higher t,
#       - Let us call the file of the queen x, of the pawn y
#
#               t    Q .             t    . Q
#               s    . .      or     s    . .
#               r    . p             r    p .
#
#                    x y                  y x

#       - The queen might go to the squares ys and yt.
#       - So these squares are under attack by one pawn in the file of the queen (x)
#       - and one pawn on the other side of the pawn. Let us call that file z

#               t    Q . .             t    . . Q
#               s    . . .      or     s    . . .
#               r    . p .             r    . p .
#
#                    x y z                  z y x

#       - The queen might go to zr and zt. It cannot be blocked, as ys and yt are empty
#       (as y has a pawn already on r)
#       - Those squares cannot be attacked from the y file, as the pawn in the y file does not
#         attack these squares
#
#       - So we must have two attacking squares in other file next to x. Contradiction.

#       So positions have
#       - at most one pawn in each file, ranks between 2 and 8
#       - one queen, not on same square as a pawn
#       - when white-to-play
#         - no queen on an attacked square
#         - no pawn on rank 8
#         - no pawn at all is possible ==> lost
#       - when black-to-play
#         - at least one pawn
#         - at most one pawn at rank 8 ==> lost
#
# How many positions are possible with white-to-play:
#   rank 2, 3, ..., 7 for each pawn and an additional "rank" for not present. That gives
#   7^8 = 5,764,801 positions
#   For the queen 64 squares when there is no pawn at all.
#   On the other hand, if we have 8 pawn and all attack different squares, then only 64 - 6*3 - 2*2 = 42 are left
#   So a lower bound is 7^8 * 42 = 242,121,642 positions

# It is too much to consider each position and store its value (win, lose, draw, how many moves).
# In cases white wins, some pawns might be irrelevant. For instance: white-to-play with pawns on a6 an b6,
# is a win. We do not need to evaluate all situation for additional pawns in the other files.
# However, additional pawns might give a win in less moves. So we will not store and optimise the number
# of moves till win.  ??????
# For any winning positions for white, it will eventually result in a promotion,
# so we should be able to generate those positions backwards.

# When is a position with white-to-play a win in n moves:
# - n=1: pawn on seventh rank not blocked by the queen
# - n>1: the is a legal pawn move, such that every legal queen-move next
#         results in a position with white to play and win in at most n-1 moves

# We can generate 'all' positions with white to play and win in n moves as follows:
# - n=1: pawn on seventh rank not blocked by the queen
# - n -> n+1: take a position winning position white to play in n moves
#             generate all possible previous positions with white to play
#               do a reverse queen move:
#                   if queen was in a file without pawns, consider with and without pawn on the place of the queen
#                   queen can come from any place, not occupied or block, but it might be attacked by one pawn (not two)
#
#               consider alternative queen moves:
#                   evaluate by taking all subsets of pawns
#               if for all alternative queen moves, there is a pawn subset such that
#                   the position is winning for white
#               then the position is losing for black

#               Then do a reverse pawn move:
#                   if the queen is attacked by one pawn consider only reverse move by that pawn
#                   if not attacked, consider all reverse pawn move, but avoid pawn on queen
#
#               Store this position is win in n+1 unless it is stored already

# ######################## POSITIONS ##########################

NOT_ON_BOARD = -1
FIRST_FILE = 1
LAST_FILE = 8
FIRST_RANK = 1
LAST_RANK = 8
files = range(FIRST_FILE, LAST_FILE + 1)
ranks = range(FIRST_RANK, LAST_RANK + 1)

WHITE = 1
BLACK = -1

WIN = 1
DRAW = 0
LOSE = -1
UNKNOWN = None


def squares():
    for f in files:
        for r in ranks:
            yield f, r


def on_board(square):
    file, rank = square
    return FIRST_FILE <= file <= LAST_FILE and FIRST_RANK <= rank <= LAST_RANK


def square_to_string(square):
    if not on_board(square):
        return "invalid"
    file, rank = square
    # noinspection SpellCheckingInspection
    return "_abcdefgh"[file] + str(rank)


def valid_pawn_rank(rank):
    return FIRST_RANK < rank <= LAST_RANK


# ######################## QUEEN MOVES ##########################

DIRECTIONS = list(product([-1, 0, 1], [-1, 0, 1]))
DIRECTIONS.remove((0, 0))


def move(queen_square, direction):
    return queen_square[0] + direction[0], queen_square[1] + direction[1]


class SetUp:
    def __init__(self):
        self.turn = WHITE
        self.queen = (FIRST_FILE, FIRST_RANK)
        self.pawns = [0] + [NOT_ON_BOARD] * 8  # we will not use self.pawns[0]

    def set_turn(self, turn):
        assert turn in [WHITE, BLACK]
        self.turn = turn

    def set_pawn(self, square):
        file, rank = square
        assert valid_pawn_rank(rank)
        self.pawns[file] = rank

    def remove_pawn(self, file):
        self.pawns[file] = NOT_ON_BOARD

    def set_queen(self, square):
        self.queen = square

    def is_valid(self):
        if self.occupied_by_pawn(self.queen):
            return False

        nb_pawns = len(list(filter(lambda f: self.pawns[f] != NOT_ON_BOARD, self.pawns)))
        nb_pawns_on_last_rank = len(list(filter(lambda f: self.pawns[f] == LAST_FILE, self.pawns)))

        if self.turn == WHITE:
            return nb_pawns > 0 and nb_pawns_on_last_rank == 0 and not self.attacked_by_pawn(self.queen)

        if self.turn == BLACK:
            return nb_pawns_on_last_rank <= 1

    def occupied_by_pawn(self, square):
        file, rank = square
        return on_board(square) and valid_pawn_rank(rank) and self.pawns[file] == rank

    def attacked_by_pawn(self, square):
        file, rank = square
        return self.occupied_by_pawn((file - 1, rank - 1)) or self.occupied_by_pawn((file + 1, rank - 1))


class Position:
    def __init__(self, *, setup=None, code=None):
        if setup:
            assert not code
            assert setup.is_valid()
            self.turn = setup.turn
            self.queen = setup.queen
            self.pawns = setup.pawns.copy()
            return
        if code:
            assert not setup
            turn, queen, pawns = code
            self.turn = turn
            self.pawns = [0] + pawns
            self.queen = queen
            return
        assert False

    def __repr__(self):
        result = f"Q: {square_to_string(self.queen)} "
        result += f"p:"
        for f in files:
            if self.pawns[f] != NOT_ON_BOARD:
                result += " " + square_to_string((f, self.pawns[f]))
        return result

    def __str__(self):
        return self.__repr__()
        result = ""
        for r in reversed(ranks):
            result += str(r)
            for f in files:
                if (f, r) == self.queen:
                    result += " Q"
                elif self.pawns[f] == r:
                    result += " p"
                else:
                    result += " ."
            result += "\n"
        result += "  a b c d e f g h\n"
        return result

    def id(self):
        return self.turn, self.queen, tuple(self.pawns[FIRST_FILE:(LAST_FILE + 1)])

    def get_pawns(self):
        for f in files:
            if self.pawns[f] != NOT_ON_BOARD:
                yield f, self.pawns[f]

    def get_copy(self):
        setup = SetUp()
        setup.set_turn(self.turn)
        for p in self.get_pawns():
            setup.set_pawn(p)
        setup.set_queen(self.queen)
        copy = Position(setup=setup)
        return copy

    def generate_next_positions(self):
        if self.turn == WHITE:
            for file in files:
                rank = self.pawns[file]
                if 2 <= rank < LAST_RANK and (file, rank + 1) != self.queen:
                    position = self.get_copy()
                    position.turn = BLACK
                    position.move_pawn_forward(file)
                    yield position
                    if rank == 2 and (file, 4) != self.queen:
                        position = self.get_copy()
                        position.turn = BLACK
                        position.move_pawn_forward_twice(file)
                        yield position

        if self.turn == BLACK:
            for d in DIRECTIONS:
                new_queen = move(self.queen, d)
                while on_board(new_queen):
                    if not self.attacked_by_pawn(new_queen):
                        position = self.get_copy()
                        position.turn = WHITE
                        position.move_queen(new_queen)
                        yield position
                    if self.occupied_by_pawn(new_queen):
                        break
                    new_queen = move(new_queen, d)

    def occupied_by_pawn(self, square):
        file, rank = square
        return on_board(square) and valid_pawn_rank(rank) and self.pawns[file] == rank

    def attacked_by_pawn(self, square):
        file, rank = square
        return self.occupied_by_pawn((file - 1, rank - 1)) or self.occupied_by_pawn((file + 1, rank - 1))

    def set_pawn(self, square):
        assert self.queen != square
        file, rank = square
        assert self.pawns[file] == NOT_ON_BOARD
        assert valid_pawn_rank(rank)
        self.pawns[file] = rank

    def remove_pawn(self, file):
        assert valid_pawn_rank(self.pawns[file])
        self.pawns[file] = NOT_ON_BOARD

    def move_pawn_forward(self, file):
        rank = self.pawns[file]
        self.remove_pawn(file)
        self.set_pawn((file, rank + 1))

    def move_pawn_forward_twice(self, file):
        assert self.pawns[file] == 2
        self.move_pawn_forward(file)
        self.move_pawn_forward(file)

    def move_queen(self, square):
        if self.occupied_by_pawn(square):
            file, rank = square
            self.remove_pawn(file)
        self.queen = square

    def evaluate(self):
        result = evaluation_store.get(self.id(), None)
        if result:
            return result

        if self.turn == WHITE:
            if all(map(lambda f: self.pawns[f] == NOT_ON_BOARD, files)):
                evaluation_store[self.id()] = LOSE
                return LOSE

        if self.turn == BLACK:
            if any(map(lambda f: self.pawns[f] == LAST_RANK, files)):
                evaluation_store[self.id()] = LOSE
                return LOSE

        best = LOSE
        nb_valid_moves = 0
        for next_pos in self.generate_next_positions():
            nb_valid_moves += 1
            best = max(best, next_pos.evaluate() * -1)
            if best == WIN:
                evaluation_store[self.id()] = WIN
                return WIN

        if nb_valid_moves == 0:
            assert self.turn == WHITE
            evaluation_store[self.id()] = DRAW
            return DRAW

        evaluation_store[self.id()] = best
        return best


# ######################## EVALUATION ##########################

evaluation_store = {}

s = SetUp()
s.set_turn(WHITE)
s.set_queen((4, 5))
p = Position(setup=s)
assert p.evaluate() == LOSE

s = SetUp()
s.set_turn(BLACK)
s.set_queen((2, 3))
s.set_pawn((2, 8))
p = Position(setup=s)
assert p.evaluate() == LOSE

s = SetUp()
s.set_turn(WHITE)
s.set_queen((4, 8))
s.set_pawn((2, 7))
p = Position(setup=s)
assert p.evaluate() == WIN

s = SetUp()
s.set_turn(WHITE)
s.set_queen((2, 8))
s.set_pawn((2, 7))
p = Position(setup=s)
assert p.evaluate() == DRAW

s = SetUp()
s.set_turn(BLACK)
s.set_queen((2, 8))
s.set_pawn((2, 6))
p = Position(setup=s)
assert p.evaluate() == WIN

s = SetUp()
s.set_pawn((5, 5))
s.set_pawn((4, 7))


for r in reversed(ranks):
    print(r, end="")
    for f in files:
        print(' ', end="")
        if s.pawns[f] == r:
            print('p', end="")
        else:
            s.set_turn(BLACK)
            s.set_queen((f, r))
            assert s.is_valid()
            eval_black_to_play = Position(setup=s).evaluate()
            s.set_turn(WHITE)
            if not s.is_valid():
                if eval_black_to_play == WIN:
                    print('+', end="")
                elif eval_black_to_play == DRAW:
                    print('0', end="")
                else:
                    print('-', end="")
            else:
                eval_white_to_play = Position(setup=s).evaluate()
                if eval_black_to_play == WIN:
                    if eval_white_to_play == WIN:
                        print('+', end="")
                    elif eval_white_to_play == DRAW:
                        print('B', end="")
                    else:
                        print('w', end="")
                elif eval_black_to_play == DRAW:
                    if eval_white_to_play == WIN:
                        print('D', end="")
                    elif eval_white_to_play == DRAW:
                        print('E', end="")
                    else:
                        print('F', end="")
                else:
                    if eval_white_to_play == WIN:
                        print('-', end="")
                    elif eval_white_to_play == DRAW:
                        print('H', end="")
                    else:
                        print('I', end="")

    print()
    # what if I change something?
exit()



evaluation_white = {}
evaluation_black = {}

for q in squares():
    p = Position(queen=q)
    code = p.id()
    evaluation_white[code] = "lose"
    evaluation_black[code] = "invalid"

assert len(evaluation_white) == len(evaluation_black) == 64

count = 0
for s in squares():
    pawn_file, pawn_rank = s
    if valid_pawn_rank(pawn_rank):
        for q in squares():
            if s != q:
                p = Position(queen=q)
                p.set_pawn(s)
                code = p.id()
                if pawn_rank == 8:
                    evaluation_white[code] = "invalid"
                    evaluation_black[code] = "lose"
                elif p.attacked_by_pawn(q):
                    evaluation_white[code] = "win"
                    evaluation_black[code] = "win"
                else:
                    evaluation_white[code] = "unknown"
                    evaluation_black[code] = "unknown"

assert len(evaluation_white) == len(evaluation_black) == 64 + 56 * 63

for code in evaluation_white:
    if evaluation_white[code] == "unknown":
        position = Position(code=code)
        move_found = False
        all_moves_are_losing = True
        some_moves_are_winning = False
        for p in position.generate_next_positions_after_a_pawn_move():
            move_found = True
            eval_black_to_play = evaluation_black[p.id()]
            if eval_black_to_play == "lose":
                some_moves_are_winning = True
                break
            if eval_black_to_play == "unknown":
                all_moves_are_losing = False
        if not move_found:
            evaluation_white[code] = "draw"
        elif some_moves_are_winning:
            evaluation_white[code] = "win"
        elif all_moves_are_losing:
            evaluation_white[code] = "lose"







exit()

evaluation = {}
new_work = []
# evaluation[position] = n means:
#   with white-to-play in position it can guarantee a win in n moves

# initialisation: evaluation[position] = 1


def init_evaluation():
    for queen_square in product(files, ranks):
        for pawn_file in files:
            pawn_square = (pawn_file, LAST_RANK - 1)
            pawn_square_next = (pawn_file, LAST_RANK)
            if queen_square not in [pawn_square, pawn_square_next]:
                position = Position(queen=queen_square)
                position.set_pawn(pawn_square)
                if not position.attacked_by_pawn(queen_square):
                    evaluation[position.id()] = 1


def get_evaluation(position):
    for nb_pawns in range(1, 9):  # 0 pawns will not win
        for selection in combinations(range(8), nb_pawns):
            less_pawns = Position(queen=position.queen)
            valid = True
            for file in selection:
                if position.pawns[file] > 0:
                    less_pawns.set_pawn((file, position.pawns[file]))
                else:
                    valid = False
                    break
            if valid:
                result = evaluation.get(less_pawns.id(), None)
                if result is not None:
                    return result
    return None


def generate_new_evaluations(position):
    found = str(position) == "Q: c3 p: c6 d4 f5"

    for prev in position.generate_prev_positions_before_a_queen_move():
        if found:
            print("prev", prev)
        best_nb = 0
        winning = True
        for _next in prev.generate_next_positions_after_a_queen_move():
            _evaluation = get_evaluation(_next)
            if found:
                print("_next", _next)
            if get_evaluation(_next) is None:
                if found:
                    print("None")
                winning = False
                break
            best_nb = max(best_nb, _evaluation)
        if winning:
            for prev_prev in prev.generate_prev_positions_before_a_pawn_move():
                if get_evaluation(prev_prev) is None:
                    prev_prev_id = prev_prev.id()
                    evaluation[prev_prev_id] = best_nb + 1
                    new_work.append(prev_prev_id)


def main():
    global new_work
    init_evaluation()
    assert len(evaluation) == 482
    new_work = list(evaluation.keys())
    while new_work:
        print(len(new_work))
        input("press enter to continue")
        work = new_work
        new_work = []
        for p in work:
            generate_new_evaluations(Position(code=p))
        for p in new_work:
            position = Position(code=p)
            print(evaluation[p], position)


main()

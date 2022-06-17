import numpy as np
import chess
import scipy.special

board = chess.Board()
matrix_board = list(str(board))[::2]


def relu(out):
    y = []

    for i in out:
        if i > 0:
            y.append([float(i)])

        else:
            y.append([0])

    print(y)
    return y


class neuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes

        self.lr = learning_rate

        self.wih = np.random.rand(self.hnodes, self.inodes) - 0.5
        self.who = np.random.rand(self.onodes, self.hnodes) - 0.5

        self.activation_function_relu = lambda x: relu(x)
        self.activation_function_softmax = lambda x: scipy.special.softmax(x)

    def train(self, board_pos, move):
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function_relu(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function_relu(final_inputs)
        final = round(np.mean(final_outputs))

        output_error = target - final

        hidden_errors = np.dot(self.who.T, output_error)

        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
        self.who += self.lr * np.dot((output_error * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))

        print(output_error)

        return output_error

    def query(self, board_pos):
        inputs = np.array(board_pos, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function_relu(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function_softmax(final_inputs)

        return final_outputs


def board_interpret():
    for i, s in enumerate(matrix_board):
        if s == '.':
            matrix_board[i] = 0
        elif s == 'r':
            matrix_board[i] = -5
        elif s == 'n':
            matrix_board[i] = -3
        elif s == 'b':
            matrix_board[i] = -3
        elif s == 'q':
            matrix_board[i] = -9
        elif s == 'k':
            matrix_board[i] = -10
        elif s == 'p':
            matrix_board[i] = -1
        elif s == 'R':
            matrix_board[i] = 5
        elif s == 'N':
            matrix_board[i] = 3
        elif s == 'B':
            matrix_board[i] = 3
        elif s == 'Q':
            matrix_board[i] = 9
        elif s == 'K':
            matrix_board[i] = 10
        elif s == 'P':
            matrix_board[i] = 1

    obs = matrix_board + [int(board.turn)]
    obs.append(len(list(board.legal_moves)))

    obs = np.array(obs)

    obs = np.pad(obs, (0, 66 - len(obs)))

    return obs


def minimax_root(depth, isMaximisingPlayer):
    moves = []
    for item in list(board.legal_moves):
        moves.append(str(item))

    best_move = -9999
    best_move_found = 0
    index = 0

    for i in np.arange(len(moves)):
        move = moves[i]
        board.push_uci(move)

        value = minimax(depth - 1, -10000, 10000, not isMaximisingPlayer)
        board.pop()

        if value >= best_move:
            best_move = value
            best_move_found = move
            index = i

    return index


def minimax(depth, alpha, beta, isMaximisingPlayer):
    if depth == 0:
        return -evaluate_board()

    moves = []
    for item in list(board.legal_moves):
        moves.append(str(item))

    if isMaximisingPlayer:
        best_move = -9999

        for i in np.arange(len(moves)):
            board.push_uci(moves[i])
            best_move = max(best_move, minimax(depth - 1, alpha, beta, not isMaximisingPlayer))
            board.pop()
            alpha = max(alpha, best_move)

            if beta <= alpha:
                return best_move

        return best_move

    else:
        best_move = 9999
        for i in np.arange(len(moves)):
            board.push_uci(moves[i])
            best_move = min(best_move, minimax(depth - 1, alpha, beta, not isMaximisingPlayer))
            board.pop()
            beta = min(beta, best_move)

            if beta <= alpha:
                return best_move

        return best_move


def calculate_best_move():
    moves = []
    for item in list(board.legal_moves):
        moves.append(str(item))

    best_move = None
    best_value = -9999

    for i in range(len(moves)):
        move = moves[i]
        board.push_uci(move)

        board_value = -evaluate_board()
        board.pop()

        if board_value > best_value:
            best_value = board_value
            best_move = move

    return best_move


def get_piece_value(piece, x, y):
    if piece == "p":
        return -10

    elif piece == "r":
        return -50

    elif piece == "n":
        return -30

    elif piece == "b":
        return -30

    elif piece == "q":
        return -90

    elif piece == "k":
        return -900

    elif piece == "P":
        return 10

    elif piece == "R":
        return 50

    elif piece == "N":
        return 30

    elif piece == "B":
        return 30

    elif piece == "Q":
        return 90

    elif piece == "K":
        return 900

    elif piece == ".":
        return 0


def evaluate_board():
    pawn = 10
    rook = 50
    knight = 30
    bishop = 30
    queen = 90
    king = 900

    wp = len(board.pieces(chess.PAWN, chess.WHITE)) * pawn
    wr = len(board.pieces(chess.ROOK, chess.WHITE)) * rook
    wn = len(board.pieces(chess.KNIGHT, chess.WHITE)) * knight
    wb = len(board.pieces(chess.BISHOP, chess.WHITE)) * bishop
    wq = len(board.pieces(chess.QUEEN, chess.WHITE)) * queen
    wk = len(board.pieces(chess.KING, chess.WHITE)) * king

    bp = len(board.pieces(chess.PAWN, chess.BLACK)) * pawn
    br = len(board.pieces(chess.ROOK, chess.BLACK)) * rook
    bn = len(board.pieces(chess.KNIGHT, chess.BLACK)) * knight
    bb = len(board.pieces(chess.BISHOP, chess.BLACK)) * bishop
    bq = len(board.pieces(chess.QUEEN, chess.BLACK)) * queen
    bk = len(board.pieces(chess.KING, chess.BLACK)) * king

    return (wp + wr + wn + wb + wq + wk) - (bp + br + bn + bb + bq + bk)


def get_best_move():
    if board.is_game_over():
        print("Game Over")

    best_move = minimax_root(depth=5, isMaximisingPlayer=True)

    return best_move


input_nodes = 66
hidden_nodes = 132
output_nodes = 218

learning_rate = 0.1

NN = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

epochs = 1

for e in range(epochs):
    print("Epoch:", e)
    y = []

    inputs = board_interpret()
    target = get_best_move()
    y.append(target)

    NN.train(inputs, y)

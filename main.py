import chess
import numpy as np

board = chess.Board()

matrix_board = np.array(list(str(board))[::2]).reshape(8, 8).tolist()


def minimax_root(depth, isMaximisingPlayer):
    moves = []
    for item in list(board.legal_moves):
        moves.append(str(item))

    best_move = -9999
    best_move_found = 0

    for i in np.arange(len(moves)):
        move = moves[i]
        board.push_uci(move)

        value = minimax(depth - 1, -10000, 10000, not isMaximisingPlayer)
        board.pop()

        if value >= best_move:
            best_move = value
            best_move_found = move

    return best_move_found


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


while True:
    print(board)
    move = input("Move: ")
    board.push_uci(move)
    print(board)
    board.push_uci(get_best_move())
    print("---------------")

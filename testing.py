import chess

board = chess.Board("8/8/3K1k2/8/8/8/8/8 w - - 0 1")

print(board.outcome().result())

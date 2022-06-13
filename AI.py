import gym
from gym import spaces
import chess
import numpy as np

board = chess.Board()


class chessEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(chessEnv, self).__init__()

        self.matrix_board = list(str(board)[::2])

        self.moves = []

        for i in list(board.legal_moves):
            self.moves.append(str(i))

        self.action_space = spaces.Discrete(1)

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(np.inf,))

    def step(self, action):
        pass

    def observation(self):

        for i, s in enumerate(self.matrix_board):
            if s == '.':
                self.matrix_board[i] = 0
            elif s == 'r':
                self.matrix_board[i] = -5
            elif s == 'n':
                self.matrix_board[i] = -3
            elif s == 'b':
                self.matrix_board[i] = -3
            elif s == 'q':
                self.matrix_board[i] = -9
            elif s == 'k':
                self.matrix_board[i] = -10
            elif s == 'p':
                self.matrix_board[i] = -1
            elif s == 'R':
                self.matrix_board[i] = 5
            elif s == 'N':
                self.matrix_board[i] = 3
            elif s == 'B':
                self.matrix_board[i] = 3
            elif s == 'Q':
                self.matrix_board[i] = 9
            elif s == 'K':
                self.matrix_board[i] = 10
            elif s == 'P':
                self.matrix_board[i] = 1

        obs = self.matrix_board + [int(board.turn)]

        # add moves
        for move in board.legal_moves:
            obs.append(int(move.from_square))
            obs.append(int(move.to_square))

        # pad with zeroes
        obs = np.pad(obs, (0, 501 - len(obs)))
        print(obs)

        return obs

    def reset(self):
        self.done = False
        # board, moves, turn
        self.observation = list(str(board)[::2])

        for move in self.moves:
            self.observation.append(move)

        return self.observation

    def render(self, mode="human"):
        print(board)

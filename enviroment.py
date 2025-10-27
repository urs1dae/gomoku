import numpy as np
from enum import IntEnum
from numpy.lib.stride_tricks import sliding_window_view


class Player(IntEnum):
    EMPTY = 0
    BLACK = 1  # first player
    WHITE = 2
    
    @property
    def piece(self):  # one-hot encoding
        vector = np.zeros(3, dtype=int)
        vector[self.value] = 1
        return vector


class GomokuEnv:
    def __init__(self, board_size: int = 15):
        self.board_size = board_size
        self.kernels = self.init_kernels()
        self.state_size = (board_size, board_size, 3)
        self.action_size = board_size * board_size

    def init_kernels(self):
        kernel_h = np.array([[1, 1, 1, 1, 1]])
        kernel_v = kernel_h.T
        kernel_d1 = np.eye(5, dtype=int)
        kernel_d2 = np.fliplr(kernel_d1)
        return [kernel_h, kernel_v, kernel_d1, kernel_d2]

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size, 3), dtype=int)
        self.current_player = Player.BLACK

    def step(self, action: tuple) -> tuple:
        x, y = action
        if self.board[x][y] != Player.EMPTY.piece:
            raise ValueError("Invalid move")
        self.board[x][y] = self.current_player.piece
        reward, done = self.check_winner()
        self.current_player = Player.WHITE if self.current_player == Player.BLACK else Player.BLACK
        return (self.board, reward, done)

    def convolve2d(self, board_layer: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        k_size = kernel.shape[0]
        padded_board = np.pad(board_layer, ((k_size - 1, k_size - 1), (k_size - 1, k_size - 1)), mode='constant')
        windows = sliding_window_view(padded_board, kernel.shape)
        conv_result = np.einsum('ijkl,kl->ij', windows, kernel)
        return conv_result

    def check_winner(self) -> tuple:
        player_board = self.board[:, :, self.current_player]
        for kernel in self.kernels:
            conv_result = self.convolve2d(player_board, kernel)
            if np.any(conv_result >= 5):
                return (1, True)
        return (0, False)

    def simulate_move(self, board: np.ndarray, action: tuple, player: Player) -> tuple:
        x, y = action
        if board[x][y] != Player.EMPTY.piece:
            raise ValueError("Invalid move")
        new_board = board.copy()
        new_board[x][y] = player.piece
        reward, done = self.check_winner_simulated(new_board, player)
        return (new_board, reward, done)

    def check_winner_simulated(self, board: np.ndarray, player: Player) -> tuple:
        player_board = board[:, :, player]
        for kernel in self.kernels:
            conv_result = self.convolve2d(player_board, kernel)
            if np.any(conv_result >= 5):
                return (1, True)
        return (0, False)
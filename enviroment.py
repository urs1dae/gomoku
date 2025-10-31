import time
import pygame
import numpy as np
from enum import IntEnum
from numpy.lib.stride_tricks import sliding_window_view


class Player(IntEnum):
    BLACK = 0  # first player
    WHITE = 1


class GomokuEnv:
    def __init__(self, board_size: int = 15):
        self.board_size = board_size
        self.kernels = self.init_kernels()
        self.state_size = (2, board_size, board_size)
        self.action_size = board_size * board_size

    def init_kernels(self):
        kernel_h = np.zeros((5, 5), dtype=int)
        kernel_h[2, :] = 1
        kernel_v = kernel_h.T
        kernel_d1 = np.eye(5, dtype=int)
        kernel_d2 = np.fliplr(kernel_d1)
        return [kernel_h, kernel_v, kernel_d1, kernel_d2]

    def draw_board(self):
        pygame.init()
        self.screen = pygame.display.set_mode((600, 600))
        pygame.display.set_caption("Gomoku")
        pygame.draw.rect(self.screen, (255, 255, 255), (0, 0, 600, 600))
        for i in range(self.board_size):
            pygame.draw.line(self.screen, (0, 0, 0), (40 + i * 40, 40), (40 + i * 40, 600 - 40), 1)
            pygame.draw.line(self.screen, (0, 0, 0), (40, 40 + i * 40), (600 - 40, 40 + i * 40), 1)
        pygame.display.flip()
    
    def draw_stone(self, x: int, y: int, player: Player):
        color = (0, 0, 0) if player == Player.BLACK else (255, 255, 255)
        pygame.draw.circle(self.screen, color, (40 + x * 40, 40 + y * 40), 15)
        pygame.display.flip()
        time.sleep(1)

    def reset(self):
        self.board = np.zeros((2, self.board_size, self.board_size), dtype=int)
        self.draw_board()
        self.current_player = Player.BLACK
        return self.board

    def step(self, action: tuple) -> tuple:
        x, y = action // self.board_size, action % self.board_size
        if self.board[self.current_player, x, y] + self.board[1 - self.current_player, x, y] != 0:
            raise ValueError("Invalid move")
        self.board[self.current_player, x, y] = 1
        self.draw_stone(x, y, self.current_player)
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
        player_board = self.board[self.current_player]
        for kernel in self.kernels:
            conv_result = self.convolve2d(player_board, kernel)
            if np.any(conv_result >= 5):
                return (1, True)
        return (0, False)

    def simulate_move(self, board: np.ndarray, action: tuple, player: Player) -> tuple:
        x, y = action // self.board_size, action % self.board_size
        if board[player, x, y] + board[1 - player, x, y] != 0 :
            raise ValueError("Invalid move")
        board[player, x, y] = 1
        reward, done = self.check_winner_simulated(board, player)
        return (board, reward, done)

    def check_winner_simulated(self, board: np.ndarray, player: Player) -> tuple:
        player_board = board[player]
        for kernel in self.kernels:
            conv_result = self.convolve2d(player_board, kernel)
            if np.any(conv_result >= 5):
                return (1, True)
        return (0, False)
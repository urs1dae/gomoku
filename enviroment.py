import time
import pygame
import numpy as np
from enum import IntEnum
from numpy.lib.stride_tricks import sliding_window_view


class Player(IntEnum):
    BLACK = 0  # first player
    WHITE = 1


class GomokuRenderer:
    def __init__(self, board_size: int = 15):

        self.board_size = board_size
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 640 + 60
        self.screen_size = (self.SCREEN_WIDTH, self.SCREEN_HEIGHT)

        self.GRID_SIZE = 640
        self.interval = 40
        self.margin = 40
        self.piece_radius = 15

    def init_board(self):
        """初始化 Pygame 窗口并绘制初始界面。"""
        pygame.init()
        self.font = pygame.font.Font(None, 30)
        self.screen = pygame.display.set_mode(self.screen_size)
        pygame.display.set_caption("Gomoku")

    def reset(self):
        """重置游戏状态到初始值，并重新绘制界面。"""
        self.board_state = np.zeros((2, self.board_size, self.board_size), dtype=int)
        self.current_turn = 1
        self.current_player = Player.BLACK
        self._draw_full_board()

    def _draw_full_board(self, show_flip=True):
        """绘制完整的棋盘、背景和文本信息，用于初始化和重置。"""
        self.screen.fill((255, 255, 255))
        pygame.draw.rect(self.screen, (244, 164, 96), (0, 0, self.GRID_SIZE, self.GRID_SIZE))

        for i in range(self.board_size):
            coord = self.margin + i * self.interval
            pygame.draw.line(self.screen, (0, 0, 0), (coord, self.margin), 
                             (coord, self.GRID_SIZE - self.margin), 1)
            pygame.draw.line(self.screen, (0, 0, 0), (self.margin, coord), 
                             (self.GRID_SIZE - self.margin, coord), 1)

        self._draw_all_pieces()
        self._draw_info_text()

        if show_flip:
            pygame.display.flip()

    def _draw_all_pieces(self):
        """根据 self.board_state 绘制所有棋子。"""
        for player in [Player.BLACK, Player.WHITE]:
            for r in range(self.board_size):
                for c in range(self.board_size):
                    if self.board_state[player, r, c] == 1:
                        x = self.margin + c * self.interval
                        y = self.margin + r * self.interval
                        self._draw_piece(x, y, player)

    def _draw_piece(self, x_pixel, y_pixel, player: Player):
        """在指定像素坐标绘制一个棋子。"""
        color = (0, 0, 0) if player == Player.BLACK else (255, 255, 255)
        pygame.draw.circle(self.screen, color, (x_pixel, y_pixel), self.piece_radius, 0)
        pygame.draw.circle(self.screen, (0, 0, 0), (x_pixel, y_pixel), self.piece_radius, 1)

    def _draw_info_text(self):
        """绘制回合数和当前落子方信息。"""
        
        turn_text = f"Turn: {self.current_turn}"
        turn_surface = self.font.render(turn_text, True, (0, 0, 0))
        self.screen.blit(turn_surface, (self.margin, self.GRID_SIZE + 20))

        player_text = "Player:"
        player_surface = self.font.render(player_text, True, (0, 0, 0))
        self.screen.blit(player_surface, (self.margin + 150, self.GRID_SIZE + 20))

        icon_center = (self.margin + 250, self.GRID_SIZE + 30)
        icon_player = self.current_player
        
        self._draw_piece(icon_center[0], icon_center[1], icon_player)

    def update_board(self, r: int, c: int, player: Player):
        """
        更新棋盘状态，绘制新棋子，并重绘所有信息。
        :param r: 棋盘行索引 (0-14)
        :param c: 棋盘列索引 (0-14)
        :param player: 落子方
        """
        self.board_state[player, r, c] = 1

        self.current_turn += 1
        self.current_player = 1 - player
        
        self._draw_full_board()
        
    def close(self):
        """关闭 Pygame 窗口。"""
        pygame.quit()


class GomokuEnv:
    def __init__(self, board_size, is_render=False):
        self.board_size = board_size
        self.kernels = self._init_kernels()
        self.state_size = (2, board_size, board_size)
        self.action_size = board_size * board_size

        self.is_render = is_render
        self.renderer = GomokuRenderer(board_size)
        
        if self.is_render:
            self.renderer.init_board()

    def reset(self):
        self.board = np.zeros((2, self.board_size, self.board_size), dtype=int)
        if self.is_render: self.renderer.reset()
        self.current_player = Player.BLACK
        return self.board

    def close(self):
        if self.is_render: self.renderer.close()

    def step(self, action, board=None, player=None):
        x, y = action // self.board_size, action % self.board_size
    
        if board is None:
            board = self.board

        real_step = False
        if player is None:
            player = self.current_player
            real_step = True

        if board[player, x, y] + board[1 - player, x, y] != 0:
            raise ValueError("Invalid move")
        board[player, x, y] = 1
        reward, done = self.check_winner(board, player)

        if real_step:
            if self.is_render: self.renderer.update_board(x, y, player)
            self.current_player = 1 - player

        return (board, reward, done)

    def _init_kernels(self):
        kernel_h = np.zeros((5, 5), dtype=int)
        kernel_h[2, :] = 1
        kernel_v = kernel_h.T
        kernel_d1 = np.eye(5, dtype=int)
        kernel_d2 = np.fliplr(kernel_d1)
        return [kernel_h, kernel_v, kernel_d1, kernel_d2]

    def _convolve2d(self, board_layer: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        k_size = kernel.shape[0]
        padded_board = np.pad(board_layer, ((k_size - 1, k_size - 1), (k_size - 1, k_size - 1)), mode='constant')
        windows = sliding_window_view(padded_board, kernel.shape)
        conv_result = np.einsum('ijkl,kl->ij', windows, kernel)
        return conv_result

    def check_winner(self, board=None, player=None):
        player_board = board[player]
        for kernel in self.kernels:
            conv_result = self._convolve2d(player_board, kernel)
            if np.any(conv_result >= 5):
                self.winner = player
                return (1, True)
        return (0, False)

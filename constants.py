# constants.py

import pygame
import numpy as np

pygame.init()

############################################
# Othello / Reversi Parametreler
############################################
ROWS = 8
COLUMNS = 8
EMPTY = 0
HUMAN = 1   # Siyah
AI = 2      # Beyaz

SQUARESIZE = 80
RADIUS = SQUARESIZE // 2 - 4

INFO_WIDTH = 360
WIDTH = COLUMNS * SQUARESIZE    # 640
HEIGHT = ROWS * SQUARESIZE      # 640
SCREEN_SIZE = (WIDTH + INFO_WIDTH, HEIGHT)

# Renkler
DARK_GREEN  = (0, 120, 0)       # Tahta zemin
BLACK       = (0, 0, 0)
WHITE       = (255, 255, 255)
GREY        = (200, 200, 200)
ORANGE      = (255, 140, 0)
RED         = (255, 0, 0)
BLUE        = (30, 110, 250)
YELLOW      = (255, 255, 0)
QUIT_RED    = (200, 0, 0)       # Quit butonu rengi

TRAIN_MINIMAX_DEPTH = 2
GAME_MINIMAX_DEPTH  = 3

# Ekran ve Fontlar
SCREEN = pygame.display.set_mode(SCREEN_SIZE)
pygame.display.set_caption("Othello / Reversi ~ AI Project")

BIG_FONT   = pygame.font.SysFont("Arial", 48, bold=True)
MED_FONT   = pygame.font.SysFont("Arial", 32)
SMALL_FONT = pygame.font.SysFont("Arial", 24)

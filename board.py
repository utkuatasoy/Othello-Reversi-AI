# board.py

import numpy as np
from .constants import ROWS, COLUMNS, EMPTY, HUMAN, AI

def create_board():
    """
    Başlangıç dizilimi (standart Othello)
    """
    board = np.zeros((ROWS, COLUMNS), dtype=int)
    board[3][3] = AI
    board[3][4] = HUMAN
    board[4][3] = HUMAN
    board[4][4] = AI
    return board

def on_board(r, c):
    """r,c koordinatının tahtada geçerli olup olmadığını döndürür."""
    return 0 <= r < ROWS and 0 <= c < COLUMNS

def get_opponent(player):
    return HUMAN if player == AI else AI

def flips_any(board, row, col, player, opponent):
    """
    Bu pozisyona player taşı koyduğumuzda, en az bir yönde
    rakip taş(lar)ını çeviriyor mu?
    """
    directions = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1)]
    for (dr, dc) in directions:
        if would_flip(board, row, col, player, opponent, dr, dc):
            return True
    return False

def would_flip(board, row, col, player, opponent, dr, dc):
    """
    Belirli (dr,dc) yönünde rakip taşları çevirme durumu var mı?
    """
    r = row + dr
    c = col + dc
    if not on_board(r, c) or board[r][c] != opponent:
        return False

    r += dr
    c += dc
    while on_board(r, c):
        if board[r][c] == EMPTY:
            return False
        if board[r][c] == player:
            return True
        r += dr
        c += dc
    return False

def valid_moves(board, player):
    """
    O anki board'da player'ın yapabileceği
    tüm geçerli (row,col) hamlelerini döndürür.
    """
    opp = get_opponent(player)
    moves = []
    for r in range(ROWS):
        for c in range(COLUMNS):
            if board[r][c] == EMPTY:
                if flips_any(board, r, c, player, opp):
                    moves.append((r,c))
    return moves

def flip_direction(board, row, col, player, opponent, dr, dc):
    """
    (dr,dc) yönünde rakip taşları çevir.
    """
    r = row + dr
    c = col + dc
    while on_board(r, c) and board[r][c] == opponent:
        r += dr
        c += dc
    if on_board(r, c) and board[r][c] == player:
        r -= dr
        c -= dc
        while (r != row) or (c != col):
            board[r][c] = player
            r -= dr
            c -= dc

def apply_move(board, row, col, player):
    """
    (row,col)'e player taşı koy,
    çevirme gereken taşları da çevir.
    """
    opponent = get_opponent(player)
    board[row][col] = player
    directions = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1)]
    for (dr, dc) in directions:
        if would_flip(board, row, col, player, opponent, dr, dc):
            flip_direction(board, row, col, player, opponent, dr, dc)

def board_full(board):
    """Board tamamen dolu mu?"""
    return not np.any(board == EMPTY)

def no_moves_left(board, player):
    """Player'ın hiç hareketi kalmadı mı?"""
    return len(valid_moves(board, player)) == 0

def is_terminal_board(board):
    """
    Oyun bitişi:
      - Tahta dolduysa
      - Ya da her iki taraf da hareket edemiyorsa
    """
    if board_full(board):
        return True
    if no_moves_left(board, HUMAN) and no_moves_left(board, AI):
        return True
    return False

def get_winner(board):
    """
    Oyun bittiğinde, siyah taş sayısı > beyaz ise "Human",
    beyaz taş sayısı > siyah ise "AI",
    eşitse None (berabere).
    """
    human_count = np.count_nonzero(board == HUMAN)
    ai_count    = np.count_nonzero(board == AI)
    if human_count > ai_count:
        return "Human"
    elif ai_count > human_count:
        return "AI"
    return None

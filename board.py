import numpy as np

# Tahta parametreleri
ROWS = 8
COLUMNS = 8
EMPTY = 0
HUMAN = 1   # Siyah
AI = 2      # Beyaz

def create_board():
    board = np.zeros((ROWS, COLUMNS), dtype=int)
    board[3][3] = AI
    board[3][4] = HUMAN
    board[4][3] = HUMAN
    board[4][4] = AI
    return board

def on_board(r, c):
    return 0 <= r < ROWS and 0 <= c < COLUMNS

def get_opponent(player):
    return HUMAN if player == AI else AI

def flips_any(board, row, col, player, opponent):
    directions = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1)]
    for dr, dc in directions:
        if would_flip(board, row, col, player, opponent, dr, dc):
            return True
    return False

def would_flip(board, row, col, player, opponent, dr, dc):
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
    opp = get_opponent(player)
    moves = []
    for r in range(ROWS):
        for c in range(COLUMNS):
            if board[r][c] == EMPTY:
                if flips_any(board, r, c, player, opp):
                    moves.append((r, c))
    return moves

def apply_move(board, row, col, player):
    opponent = get_opponent(player)
    board[row][col] = player
    directions = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1)]
    for dr, dc in directions:
        if would_flip(board, row, col, player, opponent, dr, dc):
            flip_direction(board, row, col, player, opponent, dr, dc)

def flip_direction(board, row, col, player, opponent, dr, dc):
    r = row + dr
    c = col + dc
    while on_board(r, c) and board[r][c] == opponent:
        r += dr
        c += dc
    if on_board(r, c) and board[r][c] == player:
        r -= dr
        c -= dc
        while (r, c) != (row, col):
            board[r][c] = player
            r -= dr
            c -= dc

def board_full(board):
    return not (board == EMPTY).any()

def no_moves_left(board, player):
    return len(valid_moves(board, player)) == 0

def is_terminal_board(board):
    if board_full(board):
        return True
    if no_moves_left(board, HUMAN) and no_moves_left(board, AI):
        return True
    return False

def get_winner(board):
    human_count = (board == HUMAN).sum()
    ai_count = (board == AI).sum()
    if human_count > ai_count:
        return "Human"
    elif ai_count > human_count:
        return "AI"
    return None

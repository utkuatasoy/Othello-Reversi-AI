import time, math
from board import is_terminal_board, valid_moves
from algorithms.minimax import minimax

def iterative_deepening_time_move(board, player, time_limit=1.0):
    start_time = time.time()
    best_move = None
    depth = 1   # başlangıç depthi
    while True:
        current_time = time.time()
        if current_time - start_time > time_limit:
            break
        move, score = minimax(board, depth, -math.inf, math.inf, True, player)
        if move is not None:
            best_move = move
        if is_terminal_board(board) or not valid_moves(board, player):
            break
        depth += 1
    return best_move

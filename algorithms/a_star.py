import random
from board import valid_moves, apply_move
from algorithms.minimax import evaluate_board_simple

def a_star_move(board, player):
    moves = valid_moves(board, player)
    if not moves:
        return None
    best_move = random.choice(moves)
    best_score = -999999
    for r, c in moves:
        temp = board.copy()
        apply_move(temp, r, c, player)
        score = evaluate_board_simple(temp, player)
        if score > best_score:
            best_score = score
            best_move = (r, c)
    return best_move

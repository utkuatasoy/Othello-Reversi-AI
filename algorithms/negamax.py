import math
from board import valid_moves, apply_move, is_terminal_board, get_opponent
from algorithms.minimax import evaluate_board_simple

def negamax_algorithm(board, depth, color, player, alpha, beta):
    if depth == 0 or is_terminal_board(board):
        return None, color * evaluate_board_simple(board, player)
    
    best_move = None
    value = -math.inf
    moves = valid_moves(board, player)
    
    if not moves:
        return None, -negamax_algorithm(board, depth-1, -color, get_opponent(player), -beta, -alpha)[1]
    
    for move in moves:
        temp = board.copy()
        apply_move(temp, move[0], move[1], player)
        _, score = negamax_algorithm(temp, depth-1, -color, get_opponent(player), -beta, -alpha)
        score = -score
        if score > value:
            value = score
            best_move = move
        alpha = max(alpha, score)
        if alpha >= beta:
            break
    return best_move, value

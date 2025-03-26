import math, random
from board import valid_moves, apply_move, get_opponent, is_terminal_board
from algorithms.minimax import evaluate_board_simple

# Global transposition table
transposition_table = {}

def minimax_move_ordering(board, depth, alpha, beta, maximizingPlayer, player_max):
    if depth == 0 or is_terminal_board(board):
        return None, evaluate_board_simple(board, player_max)
    
    current_player = player_max if maximizingPlayer else get_opponent(player_max)
    moves = valid_moves(board, current_player)
    if not moves:
        return None, minimax_move_ordering(board, depth - 1, alpha, beta, not maximizingPlayer, player_max)[1]
    
    board_key = (tuple(board.flatten()), current_player)
    if board_key in transposition_table:
        trans_move = transposition_table[board_key]
        if trans_move in moves:
            moves.remove(trans_move)
            moves = [trans_move] + moves
    else:
        def move_score(move):
            temp = board.copy()
            apply_move(temp, move[0], move[1], current_player)
            return evaluate_board_simple(temp, player_max)
        if maximizingPlayer:
            moves = sorted(moves, key=lambda m: move_score(m), reverse=True)
        else:
            moves = sorted(moves, key=lambda m: move_score(m))
    
    best_move = None
    if maximizingPlayer:
        value = -math.inf
        for move in moves:
            temp = board.copy()
            apply_move(temp, move[0], move[1], current_player)
            _, score = minimax_move_ordering(temp, depth - 1, alpha, beta, not maximizingPlayer, player_max)
            if score > value:
                value = score
                best_move = move
            alpha = max(alpha, value)
            if alpha >= beta:
                break
    else:
        value = math.inf
        for move in moves:
            temp = board.copy()
            apply_move(temp, move[0], move[1], current_player)
            _, score = minimax_move_ordering(temp, depth - 1, alpha, beta, not maximizingPlayer, player_max)
            if score < value:
                value = score
                best_move = move
            beta = min(beta, value)
            if alpha >= beta:
                break
    transposition_table[board_key] = best_move
    return best_move, value

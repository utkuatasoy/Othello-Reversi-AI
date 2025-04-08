import math
import numpy as np
from board import valid_moves, apply_move, get_opponent, is_terminal_board
from algorithms.minimax import evaluate_board_simple

def coin_parity(board, player): # tahtadaki taşların sayısı oyuncu vs. ai
    opp = get_opponent(player)
    player_count = (board == player).sum()
    opp_count = (board == opp).sum()
    if player_count + opp_count == 0:
        return 0
    return 100 * (player_count - opp_count) / (player_count + opp_count)

def mobility(board, player):    # taş farkını % lik gönder
    opp = get_opponent(player)
    player_moves = len(valid_moves(board, player))
    opp_moves = len(valid_moves(board, opp))
    if player_moves + opp_moves == 0:
        return 0
    return 100 * (player_moves - opp_moves) / (player_moves + opp_moves)

def corners_captured(board, player):    # köşeleri kontrol et
    opp = get_opponent(player)
    ROWS, COLUMNS = board.shape
    corners = [(0, 0), (0, COLUMNS-1), (ROWS-1, 0), (ROWS-1, COLUMNS-1)]
    player_corners = sum(1 for (r, c) in corners if board[r][c] == player)
    opp_corners = sum(1 for (r, c) in corners if board[r][c] == opp)
    if player_corners + opp_corners == 0:
        return 0
    return 100 * (player_corners - opp_corners) / (player_corners + opp_corners)

def stability(board, player):   # taşlar ne kadar sağlam
    opp = get_opponent(player)
    stable_weight = 1.0
    semi_stable_weight = 0.5
    unstable_weight = 0.1

    def player_stability(p):
        stable = 0
        semi_stable = 0
        unstable = 0
        ROWS, COLUMNS = board.shape
        for r in range(ROWS):
            for c in range(COLUMNS):
                if board[r][c] == p:
                    if (r, c) in [(0, 0), (0, COLUMNS-1), (ROWS-1, 0), (ROWS-1, COLUMNS-1)]:
                        stable += 1
                    elif r == 0 or r == ROWS-1 or c == 0 or c == COLUMNS-1:
                        semi_stable += 1
                    else:
                        unstable += 1
        return stable_weight * stable + semi_stable_weight * semi_stable + unstable_weight * unstable

    player_val = player_stability(player)
    opp_val = player_stability(opp)
    total = player_val + opp_val
    if total == 0:
        return 0
    return 100 * (player_val - opp_val) / total

def evaluate_board_advanced(board, player):
    cp = coin_parity(board, player)
    mob = mobility(board, player)
    cor = corners_captured(board, player)
    stab = stability(board, player)
    return 0.25 * cp + 0.25 * mob + 0.3 * cor + 0.2 * stab

def minimax_advanced(board, depth, alpha, beta, maximizingPlayer, player_max):
    if depth == 0 or is_terminal_board(board):
        return None, evaluate_board_advanced(board, player_max)
    
    current_player = player_max if maximizingPlayer else get_opponent(player_max)
    moves = valid_moves(board, current_player)
    if not moves:
        return None, minimax_advanced(board, depth-1, alpha, beta, not maximizingPlayer, player_max)[1]
    
    if maximizingPlayer:
        value = -math.inf
        best_move = moves[0]
        for move in moves:
            temp = board.copy()
            apply_move(temp, move[0], move[1], player_max)
            new_score = minimax_advanced(temp, depth-1, alpha, beta, False, player_max)[1]
            if new_score > value:
                value = new_score
                best_move = move
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return best_move, value
    else:
        value = math.inf
        best_move = moves[0]
        opp = get_opponent(player_max)
        for move in moves:
            temp = board.copy()
            apply_move(temp, move[0], move[1], opp)
            new_score = minimax_advanced(temp, depth-1, alpha, beta, True, player_max)[1]
            if new_score < value:
                value = new_score
                best_move = move
            beta = min(beta, value)
            if alpha >= beta:
                break
        return best_move, value

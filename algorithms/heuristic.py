# algorithms/heuristic.py
import numpy as np
from config import HUMAN, AI

def evaluate_board_simple(board, player):
    opponent = HUMAN if player == AI else AI
    player_count = np.count_nonzero(board == player)
    opp_count = np.count_nonzero(board == opponent)
    return player_count - opp_count

def coin_parity(board, player):
    opponent = HUMAN if player == AI else AI
    player_count = np.count_nonzero(board == player)
    opp_count = np.count_nonzero(board == opponent)
    if player_count + opp_count == 0:
        return 0
    return 100 * (player_count - opp_count) / (player_count + opp_count)

def mobility(board, player):
    from board import valid_moves
    opponent = HUMAN if player == AI else AI
    player_moves = len(valid_moves(board, player))
    opp_moves = len(valid_moves(board, opponent))
    if player_moves + opp_moves == 0:
        return 0
    return 100 * (player_moves - opp_moves) / (player_moves + opp_moves)

def corners_captured(board, player):
    opponent = HUMAN if player == AI else AI
    corners = [(0,0), (0, board.shape[1]-1), (board.shape[0]-1, 0), (board.shape[0]-1, board.shape[1]-1)]
    player_corners = sum(1 for (r, c) in corners if board[r][c] == player)
    opp_corners = sum(1 for (r, c) in corners if board[r][c] == opponent)
    if player_corners + opp_corners == 0:
        return 0
    return 100 * (player_corners - opp_corners) / (player_corners + opp_corners)

def stability(board, player):
    opponent = HUMAN if player == AI else AI
    stable_weight = 1.0
    semi_stable_weight = 0.5
    unstable_weight = 0.1
    def player_stability(p):
        stable = 0
        semi_stable = 0
        unstable = 0
        rows, cols = board.shape
        for r in range(rows):
            for c in range(cols):
                if board[r][c] == p:
                    if (r, c) in [(0,0), (0,cols-1), (rows-1,0), (rows-1,cols-1)]:
                        stable += 1
                    elif r == 0 or r == rows-1 or c == 0 or c == cols-1:
                        semi_stable += 1
                    else:
                        unstable += 1
        return stable_weight * stable + semi_stable_weight * semi_stable + unstable_weight * unstable
    player_val = player_stability(player)
    opp_val = player_stability(opponent)
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

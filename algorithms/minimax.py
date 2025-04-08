import math, random
from board import is_terminal_board, valid_moves, apply_move, get_opponent

def evaluate_board_simple(board, player):   # tahtadaki taşlara bakıp skor
    opp = get_opponent(player)
    player_count = (board == player).sum()
    opp_count = (board == opp).sum()
    return player_count - opp_count

def minimax(board, depth, alpha, beta, maximizingPlayer, player_max):  # minimax algoritması
    if depth == 0 or is_terminal_board(board):
        return None, evaluate_board_simple(board, player_max)   # bitti mi?
    
    moves = valid_moves(board, player_max if maximizingPlayer else get_opponent(player_max))
    if not moves:   # geçerli hamle yoksa
        # pas durumu
        if maximizingPlayer:
            return None, minimax(board, depth-1, alpha, beta, False, player_max)[1]
        else:
            return None, minimax(board, depth-1, alpha, beta, True, player_max)[1]
    
    if maximizingPlayer:
        value = -math.inf
        best_move = random.choice(moves)
        for r, c in moves:
            temp = board.copy()
            apply_move(temp, r, c, player_max)
            new_score = minimax(temp, depth-1, alpha, beta, False, player_max)[1]
            if new_score > value:
                value = new_score
                best_move = (r, c)
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return best_move, value
    else:   # minimizing player
        # opponent player
        value = math.inf
        best_move = random.choice(moves)
        opp = get_opponent(player_max)
        for r, c in moves:
            temp = board.copy()
            apply_move(temp, r, c, opp)
            new_score = minimax(temp, depth-1, alpha, beta, True, player_max)[1]
            if new_score < value:
                value = new_score
                best_move = (r, c)
            beta = min(beta, value)
            if alpha >= beta:
                break
        return best_move, value

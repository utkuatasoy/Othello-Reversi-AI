# ai_move.py

import math
import random
import time
import pickle
import os
import pygame
import numpy as np

from .board import (
    valid_moves, apply_move, is_terminal_board,
    get_opponent, get_winner, no_moves_left, create_board
)
from .constants import (
    HUMAN, AI, GAME_MINIMAX_DEPTH,
    TRAIN_MINIMAX_DEPTH
)
from .draw import draw_board_training_visual
from .board import board_full, valid_moves, on_board, would_flip

############################################
# Q-Table (Global)
############################################
q_table = {}

############################################
# Basit Değer Hesaplama (evaluate)
############################################
def evaluate_board_simple(board, player):
    opp = get_opponent(player)
    player_count = np.count_nonzero(board == player)
    opp_count    = np.count_nonzero(board == opp)
    return player_count - opp_count

############################################
# Minimax
############################################
def minimax(board, depth, alpha, beta, maximizingPlayer, player_max=AI):
    if depth == 0 or is_terminal_board(board):
        return None, evaluate_board_simple(board, player_max)

    vm = valid_moves(board, player_max if maximizingPlayer else get_opponent(player_max))
    if not vm:
        # Pas durumu
        if maximizingPlayer:
            return None, minimax(board, depth-1, alpha, beta, False, player_max)[1]
        else:
            return None, minimax(board, depth-1, alpha, beta, True, player_max)[1]

    if maximizingPlayer:
        value = -math.inf
        best_move = random.choice(vm)
        for (r,c) in vm:
            temp = board.copy()
            apply_move(temp, r, c, player_max)
            new_score = minimax(temp, depth-1, alpha, beta, False, player_max)[1]
            if new_score > value:
                value = new_score
                best_move = (r,c)
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return best_move, value
    else:
        opp = get_opponent(player_max)
        value = math.inf
        best_move = random.choice(vm)
        for (r,c) in vm:
            temp = board.copy()
            apply_move(temp, r, c, opp)
            new_score = minimax(temp, depth-1, alpha, beta, True, player_max)[1]
            if new_score < value:
                value= new_score
                best_move= (r,c)
            beta= min(beta, value)
            if alpha>=beta:
                break
        return best_move, value

############################################
# A* basit (en yüksek evaluate_board_simple skoru)
############################################
def a_star_move(board, player=AI):
    vm = valid_moves(board, player)
    if not vm:
        return None
    best_move = random.choice(vm)
    best_score= -999999
    for (r,c) in vm:
        temp= board.copy()
        apply_move(temp, r, c, player)
        sc = evaluate_board_simple(temp, player)
        if sc > best_score:
            best_score = sc
            best_move  = (r,c)
    return best_move

############################################
# MCTS
############################################

class MCTSNode:
    def __init__(self, board, parent=None, current_player=AI, last_move=None):
        self.board = board
        self.parent = parent
        self.current_player = current_player
        self.last_move = last_move

        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = valid_moves(board, current_player)

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def ucb_score(self, c=1.4142):
        if self.visits == 0:
            return float('inf')
        return (self.wins/self.visits) + c * math.sqrt(math.log(self.parent.visits)/ self.visits)

    def select_child(self):
        return max(self.children, key=lambda c: c.ucb_score())

    def expand(self):
        if not self.untried_moves:
            # Pas
            temp = self.board.copy()
            np   = get_opponent(self.current_player)
            child = MCTSNode(temp, parent=self, current_player=np, last_move=None)
            self.children.append(child)
            return child

        move = random.choice(self.untried_moves)
        temp = self.board.copy()
        apply_move(temp, move[0], move[1], self.current_player)
        np = get_opponent(self.current_player)
        child = MCTSNode(temp, parent=self, current_player=np, last_move=move)
        self.children.append(child)
        self.untried_moves.remove(move)
        return child

    def rollout(self):
        temp = self.board.copy()
        cp   = self.current_player
        while not is_terminal_board(temp):
            mv = valid_moves(temp, cp)
            if not mv:
                cp = get_opponent(cp)
                if not valid_moves(temp, cp):
                    break
                continue
            chosen = random.choice(mv)
            apply_move(temp, chosen[0], chosen[1], cp)
            cp = get_opponent(cp)

        w = get_winner(temp)
        if w == "AI":
            return 1
        elif w == "Human":
            return -1
        return 0

    def backpropagate(self, result):
        self.visits += 1
        self.wins   += result
        if self.parent:
            self.parent.backpropagate(result)

def mcts_move(board, player=AI, simulations=50):
    vm = valid_moves(board, player)
    if not vm:
        return None
    root = MCTSNode(board.copy(), None, player)
    for _ in range(simulations):
        node = root
        # Selection
        while node.is_fully_expanded() and node.children:
            node = node.select_child()
        # Expansion
        if not is_terminal_board(node.board):
            node = node.expand()
        # Rollout
        result = node.rollout()
        # Backprop
        node.backpropagate(result)

    if not root.children:
        return None
    best_child = max(root.children, key=lambda c: c.visits)
    return best_child.last_move

############################################
# Q-Learning
############################################
def get_state_key(board, player):
    return (tuple(board.flatten()), player)

def get_best_q_action(board, player):
    vm = valid_moves(board, player)
    if not vm:
        return None
    st = get_state_key(board, player)
    if st not in q_table:
        return random.choice(vm)
    best_mv = None
    best_val= -999999
    for mv in vm:
        val = q_table[st].get(mv, 0.0)
        if val > best_val:
            best_val = val
            best_mv  = mv
    if best_mv is None:
        return random.choice(vm)
    return best_mv

def q_learning_move(board, player=AI):
    return get_best_q_action(board, player)

############################################
# Negamax
############################################
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

############################################
# Gelişmiş Heuristik
############################################
def coin_parity(board, player):
    opp = get_opponent(player)
    player_count = np.count_nonzero(board == player)
    opp_count    = np.count_nonzero(board == opp)
    if player_count + opp_count == 0:
        return 0
    return 100 * (player_count - opp_count) / (player_count + opp_count)

def mobility(board, player):
    opp = get_opponent(player)
    player_moves = len(valid_moves(board, player))
    opp_moves    = len(valid_moves(board, opp))
    if player_moves + opp_moves == 0:
        return 0
    return 100 * (player_moves - opp_moves) / (player_moves + opp_moves)

def corners_captured(board, player):
    opp = get_opponent(player)
    corners = [(0,0), (0,board.shape[1]-1), (board.shape[0]-1,0), (board.shape[0]-1, board.shape[1]-1)]
    player_corners = sum(1 for (r,c) in corners if board[r][c] == player)
    opp_corners    = sum(1 for (r,c) in corners if board[r][c] == opp)
    if player_corners + opp_corners == 0:
        return 0
    return 100 * (player_corners - opp_corners) / (player_corners + opp_corners)

def stability(board, player):
    opp = get_opponent(player)
    stable_weight     = 1.0
    semi_stable_weight= 0.5
    unstable_weight   = 0.1

    def player_stability(p):
        stable = 0
        semi   = 0
        unstab = 0
        rows, cols = board.shape
        for r in range(rows):
            for c in range(cols):
                if board[r][c] == p:
                    if (r, c) in [(0,0), (0,cols-1), (rows-1,0), (rows-1,cols-1)]:
                        stable += 1
                    elif r == 0 or r == rows-1 or c == 0 or c == cols-1:
                        semi += 1
                    else:
                        unstab += 1
        return stable_weight*stable + semi_stable_weight*semi + unstable_weight*unstab

    player_val = player_stability(player)
    opp_val    = player_stability(opp)
    total = player_val + opp_val
    if total == 0:
        return 0
    return 100 * (player_val - opp_val) / total

def evaluate_board_advanced(board, player):
    cp  = coin_parity(board, player)
    mob = mobility(board, player)
    cor = corners_captured(board, player)
    stab= stability(board, player)
    # Ağırlıklar: parity %25, mobility %25, corners %30, stability %20
    return 0.25*cp + 0.25*mob + 0.3*cor + 0.2*stab

def minimax_advanced(board, depth, alpha, beta, maximizingPlayer, player_max=AI):
    if depth == 0 or is_terminal_board(board):
        return None, evaluate_board_advanced(board, player_max)

    current_player = player_max if maximizingPlayer else get_opponent(player_max)
    moves = valid_moves(board, current_player)
    if not moves:
        return None, minimax_advanced(board, depth-1, alpha, beta, not maximizingPlayer, player_max)[1]

    if maximizingPlayer:
        value = -math.inf
        best_move = random.choice(moves)
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
        best_move = random.choice(moves)
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

############################################
# Iterative Deepening (Time Sınırlı)
############################################
def iterative_deepening_time_move(board, player=AI, time_limit=1.0):
    start_time = time.time()
    best_move  = None
    depth      = 1
    import math

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

############################################
# Move Ordering
############################################
def minimax_move_ordering(board, depth, alpha, beta, maximizingPlayer, player_max=AI):
    if depth == 0 or is_terminal_board(board):
        return None, evaluate_board_simple(board, player_max)

    current_player = player_max if maximizingPlayer else get_opponent(player_max)
    moves = valid_moves(board, current_player)
    if not moves:
        return None, minimax_move_ordering(board, depth - 1, alpha, beta, not maximizingPlayer, player_max)[1]

    # Transposition table (global)
    global transposition_table
    if 'transposition_table' not in globals():
        transposition_table = {}

    board_key = (tuple(board.flatten()), current_player)
    trans_move = transposition_table.get(board_key)

    # Move-ordering
    if trans_move is not None and trans_move in moves:
        moves.remove(trans_move)
        moves = [trans_move] + moves
    else:
        def move_score(m):
            temp = board.copy()
            apply_move(temp, m[0], m[1], current_player)
            return evaluate_board_simple(temp, player_max)

        if maximizingPlayer:
            moves = sorted(moves, key=lambda m: move_score(m), reverse=True)
        else:
            moves = sorted(moves, key=lambda m: move_score(m))

    best_move = None
    if maximizingPlayer:
        value = -math.inf
        for mv in moves:
            temp = board.copy()
            apply_move(temp, mv[0], mv[1], current_player)
            _, score = minimax_move_ordering(temp, depth - 1, alpha, beta, not maximizingPlayer, player_max)
            if score > value:
                value = score
                best_move = mv
            alpha = max(alpha, value)
            if alpha >= beta:
                break
    else:
        value = math.inf
        for mv in moves:
            temp = board.copy()
            apply_move(temp, mv[0], mv[1], current_player)
            _, score = minimax_move_ordering(temp, depth - 1, alpha, beta, not maximizingPlayer, player_max)
            if score < value:
                value = score
                best_move = mv
            beta = min(beta, value)
            if alpha >= beta:
                break

    transposition_table[board_key] = best_move
    return best_move, value

############################################
# AI Hamlesi (Zorluk da eklenmiş)
############################################
def difficulty_to_random_prob(difficulty):
    if difficulty == "Easy":
        return 0.75
    elif difficulty == "Medium":
        return 0.50
    elif difficulty == "Hard":
        return 0.25
    else:
        return 0.0

def ai_move(board, let_me_win=False, algorithm="Min-Max with Alpha-Beta Pruning", difficulty="Easy"):
    if let_me_win:
        # Kazandırmak için AI hamle yapmasın => None (pas)
        return None

    vm = valid_moves(board, AI)
    if not vm:
        return None

    rnd_prob = difficulty_to_random_prob(difficulty)
    if random.random() < rnd_prob:
        return random.choice(vm)
    else:
        if algorithm == "A* algorithm":
            return a_star_move(board, AI)
        elif algorithm == "Monte Carlo Tree Search (MCTS)":
            return mcts_move(board, AI, simulations=50)
        elif algorithm == "Min-Max with Alpha-Beta Pruning":
            best_mv, _ = minimax(board, GAME_MINIMAX_DEPTH, -math.inf, math.inf, True, AI)
            return best_mv
        elif algorithm == "Q-Learning":
            return q_learning_move(board, AI)
        elif algorithm == "Advanced Heuristic Based Search":
            best_mv, _ = minimax_advanced(board, GAME_MINIMAX_DEPTH, -math.inf, math.inf, True, AI)
            return best_mv
        elif algorithm == "Negamax Algorithm":
            best_mv, _ = negamax_algorithm(board, GAME_MINIMAX_DEPTH, 1, AI, -math.inf, math.inf)
            return best_mv
        elif algorithm == "Iterative Deepening with Time Constraint":
            return iterative_deepening_time_move(board, AI, time_limit=1.0)
        elif algorithm == "Move Ordering":
            best_mv, _ = minimax_move_ordering(board, GAME_MINIMAX_DEPTH, -math.inf, math.inf, True, AI)
            return best_mv
        else:
            best_mv, _ = minimax(board, GAME_MINIMAX_DEPTH, -math.inf, math.inf, True, AI)
            return best_mv

############################################
# Q-Learning rakip (Minimax) - Eğitim
############################################
def minimax_training_opponent(board):
    from math import inf
    vm = valid_moves(board, HUMAN)
    if not vm:
        return None
    best_mv, _ = minimax(board, TRAIN_MINIMAX_DEPTH, -inf, inf, False, AI)
    return best_mv

############################################
# Görsel Q-Learning Eğitimi
############################################
def train_q_learning_visual_custom(
    episodes=10, alpha=0.1, gamma=0.95,
    epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.99
):
    global q_table
    ep = 0
    clock = pygame.time.Clock()

    print("[TRAIN VISUAL] Starting Q-Learning... Press Q to stop early.")

    while ep < episodes:
        board = create_board()
        game_over = False
        import random
        turn = random.choice([HUMAN, AI])

        while not game_over:
            draw_board_training_visual(board, turn, game_over)
            pygame.time.wait(30)
            clock.tick(30)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    import sys
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        with open("q_table_othello.pkl", "wb") as f:
                            pickle.dump(q_table, f)
                        print(f"[TRAIN VISUAL] Early exit. {ep} episodes done.")
                        return

            if turn == AI:
                st = (tuple(board.flatten()), AI)
                if st not in q_table:
                    q_table[st] = {}

                moves = valid_moves(board, AI)
                if not moves:
                    turn = HUMAN
                    if not valid_moves(board, HUMAN):
                        game_over = True
                    continue

                if random.random() < epsilon:
                    action = random.choice(moves)
                else:
                    best_val= -999999
                    action = None
                    for mv in moves:
                        val = q_table[st].get(mv, 0.0)
                        if val > best_val:
                            best_val = val
                            action = mv

                row, col = action
                apply_move(board, row, col, AI)
                reward = 0.0
                if is_terminal_board(board):
                    w = get_winner(board)
                    if w == "AI":
                        reward = 1.0
                    elif w == "Human":
                        reward = -1.0
                    game_over = True

                new_st = (tuple(board.flatten()), HUMAN)
                if new_st not in q_table:
                    q_table[new_st] = {}

                old_q = q_table[st].get(action, 0.0)
                if not game_over:
                    fut_moves = valid_moves(board, HUMAN)
                    if fut_moves:
                        max_future = max(q_table[new_st].get(mv, 0.0) for mv in fut_moves)
                    else:
                        max_future = 0
                else:
                    max_future = 0

                new_q = old_q + alpha * (reward + gamma*max_future - old_q)
                q_table[st][action] = new_q

                if not game_over:
                    turn = HUMAN
            else:
                # Rakip Human değil, Minimax
                mv = valid_moves(board, HUMAN)
                if not mv:
                    turn = AI
                    if not valid_moves(board, AI):
                        game_over = True
                    continue
                best_mv = minimax_training_opponent(board)
                if best_mv is None:
                    best_mv = random.choice(mv)
                r, c = best_mv
                apply_move(board, r, c, HUMAN)
                if is_terminal_board(board):
                    w = get_winner(board)
                    game_over = True
                else:
                    turn = AI

        ep += 1
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        print(f"[TRAIN VISUAL] Episode {ep}/{episodes} done. Epsilon={epsilon:.3f}")

    with open("q_table_othello.pkl","wb") as f:
        pickle.dump(q_table, f)
    print("[TRAIN VISUAL] All training done. q_table_othello.pkl saved.")

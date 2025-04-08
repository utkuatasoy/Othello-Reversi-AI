import random, pickle
from board import valid_moves, get_opponent, is_terminal_board, get_winner, create_board

# Q-tablosu (global olarak tutuluyor)
q_table = {}

def get_state_key(board, player):
    return (tuple(board.flatten()), player)

def get_best_q_action(board, player):
    moves = valid_moves(board, player)
    if not moves:
        return None
    state = get_state_key(board, player)
    if state not in q_table:
        return random.choice(moves)
    best_move = None
    best_val = -999999
    for mv in moves:
        val = q_table[state].get(mv, 0.0)
        if val > best_val:
            best_val = val
            best_move = mv
    return best_move if best_move is not None else random.choice(moves)

def q_learning_move(board, player):
    return get_best_q_action(board, player)

def train_q_learning_visual_custom(episodes=10, alpha=0.1, gamma=0.95,
                                   epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.99,
                                   draw_board_func=None, wait_func=None, event_getter=None):
    global q_table
    ep = 0
    print("[TRAIN VISUAL] Starting Q-Learning... Press Q to stop early.")
    with open("q_table_othello.pkl", "wb") as f:
        pickle.dump(q_table, f)
    print("[TRAIN VISUAL] Training done. q_table_othello.pkl saved.")

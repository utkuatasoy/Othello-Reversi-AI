import pygame, sys, math, random, time, csv, os, numpy as np, pickle
from board import create_board, valid_moves, apply_move, is_terminal_board, get_winner, HUMAN, AI, ROWS, COLUMNS, EMPTY
from algorithms.minimax import minimax
from algorithms.a_star import a_star_move
from algorithms.mcts import mcts_move
from algorithms.negamax import negamax_algorithm
from algorithms.advanced import minimax_advanced
from algorithms.iterative_deepening import iterative_deepening_time_move
from algorithms.move_ordering import minimax_move_ordering

# UI ve oyun parametreleri
SQUARESIZE = 80
RADIUS = SQUARESIZE // 2 - 4
WIDTH = COLUMNS * SQUARESIZE
HEIGHT = ROWS * SQUARESIZE
INFO_WIDTH = 360
SCREEN_SIZE = (WIDTH + INFO_WIDTH, HEIGHT)

# Renkler
DARK_GREEN  = (0,120,0)
BLACK       = (0,0,0)
WHITE       = (255,255,255)
GREY        = (200,200,200)
ORANGE      = (255,140,0)
RED         = (255,0,0)
BLUE        = (30,110,250)
YELLOW      = (255,255,0)
QUIT_RED    = (200,0,0)

TRAIN_MINIMAX_DEPTH = 2
GAME_MINIMAX_DEPTH = 3

pygame.init()
SCREEN = pygame.display.set_mode(SCREEN_SIZE)
pygame.display.set_caption("Othello / Reversi ~ AI Project")

BIG_FONT   = pygame.font.SysFont("Arial", 48, bold=True)
MED_FONT   = pygame.font.SysFont("Arial", 32)
SMALL_FONT = pygame.font.SysFont("Arial", 24)

############################################
# Board Çizme Fonksiyonları
############################################
def draw_whole_board(board, turn, game_over, ai_last_move_time, avg_time, total_moves,
                     ai_total_time=0.0, username="", algorithm="", difficulty="Easy"):
    SCREEN.fill(BLACK)
    # Tahta kareleri
    for r in range(ROWS):
        for c in range(COLUMNS):
            x_pos = c * SQUARESIZE
            y_pos = r * SQUARESIZE
            pygame.draw.rect(SCREEN, DARK_GREEN, (x_pos, y_pos, SQUARESIZE, SQUARESIZE))
            pygame.draw.rect(SCREEN, WHITE, (x_pos, y_pos, SQUARESIZE, SQUARESIZE), 1)
    # Taşlar
    for r in range(ROWS):
        for c in range(COLUMNS):
            piece = board[r][c]
            if piece != 0:
                color = BLACK if piece == HUMAN else WHITE
                cx = c * SQUARESIZE + SQUARESIZE//2
                cy = r * SQUARESIZE + SQUARESIZE//2
                pygame.draw.circle(SCREEN, color, (cx, cy), RADIUS)
    # İnsan hamleleri
    if not game_over and turn == HUMAN:
        moves = valid_moves(board, HUMAN)
        for (rr, cc) in moves:
            cx = cc * SQUARESIZE + SQUARESIZE//2
            cy = rr * SQUARESIZE + SQUARESIZE//2
            pygame.draw.circle(SCREEN, (0,200,0), (cx, cy), 8)
    # Sağ panel
    pygame.draw.rect(SCREEN, GREY, (WIDTH, 0, INFO_WIDTH, HEIGHT))
    user_text = SMALL_FONT.render(f"Player: {username}", True, BLACK)
    SCREEN.blit(user_text, (WIDTH+20, 10))
    algo_text = SMALL_FONT.render(f"Algorithm: {algorithm}", True, BLACK)
    SCREEN.blit(algo_text, (WIDTH+20, 45))
    diff_text = SMALL_FONT.render(f"Difficulty: {difficulty}", True, BLACK)
    SCREEN.blit(diff_text, (WIDTH+20, 80))
    if not game_over:
        if turn == HUMAN:
            turn_txt = MED_FONT.render("Your Turn (Black)", True, BLACK)
        else:
            turn_txt = MED_FONT.render("AI's Turn (White)", True, BLACK)
    else:
        turn_txt = MED_FONT.render("Game Over", True, BLACK)
    SCREEN.blit(turn_txt, (WIDTH+20, 130))
    time_txt = SMALL_FONT.render(f"AI Last Move Time: {ai_last_move_time:.4f}s", True, BLACK)
    SCREEN.blit(time_txt, (WIDTH+20, 170))
    if avg_time is not None:
        avg_txt = SMALL_FONT.render(f"Avg Time: {avg_time:.4f}s", True, BLACK)
        SCREEN.blit(avg_txt, (WIDTH+20, 195))
    moves_txt = SMALL_FONT.render(f"Total Moves: {total_moves}", True, BLACK)
    SCREEN.blit(moves_txt, (WIDTH+20, 220))
    ai_total_txt = SMALL_FONT.render(f"AI Total Time: {ai_total_time:.4f}s", True, BLACK)
    SCREEN.blit(ai_total_txt, (WIDTH+20, 245))
    hc = (board == HUMAN).sum()
    ac = (board == AI).sum()
    cnt_txt = SMALL_FONT.render(f"Black: {hc} | White: {ac}", True, BLACK)
    SCREEN.blit(cnt_txt, (WIDTH+20, 290))
    quit_button = pygame.Rect(WIDTH+20, 340, 120, 40)
    pygame.draw.rect(SCREEN, QUIT_RED, quit_button)
    quit_surf = SMALL_FONT.render("QUIT (Q)", True, WHITE)
    quit_rect = quit_surf.get_rect(center=quit_button.center)
    SCREEN.blit(quit_surf, quit_rect)
    pygame.display.update()

############################################
# Animasyon Fonksiyonları
############################################
def animate_piece_place(board, row, col, player,
                        turn, game_over, ai_last_move_time, avg_time, total_moves,
                        ai_total_time=0.0, username="", algorithm="", difficulty="Easy"):
    clock = pygame.time.Clock()
    color = BLACK if player == HUMAN else WHITE
    steps = 12
    for step in range(1, steps+1):
        SCREEN.fill(BLACK)
        for rr in range(ROWS):
            for cc in range(COLUMNS):
                x_pos = cc * SQUARESIZE
                y_pos = rr * SQUARESIZE
                pygame.draw.rect(SCREEN, DARK_GREEN, (x_pos, y_pos, SQUARESIZE, SQUARESIZE))
                pygame.draw.rect(SCREEN, WHITE, (x_pos, y_pos, SQUARESIZE, SQUARESIZE), 1)
        for rr in range(ROWS):
            for cc in range(COLUMNS):
                piece = board[rr][cc]
                if piece != 0:
                    clr = BLACK if piece == HUMAN else WHITE
                    cx = cc * SQUARESIZE + SQUARESIZE//2
                    cy = rr * SQUARESIZE + SQUARESIZE//2
                    pygame.draw.circle(SCREEN, clr, (cx, cy), RADIUS)
        cx_new = col * SQUARESIZE + SQUARESIZE//2
        cy_new = row * SQUARESIZE + SQUARESIZE//2
        current_radius = int(RADIUS * (step / steps))
        pygame.draw.circle(SCREEN, color, (cx_new, cy_new), current_radius)
        pygame.draw.rect(SCREEN, GREY, (WIDTH, 0, INFO_WIDTH, HEIGHT))
        user_text = SMALL_FONT.render(f"Player: {username}", True, BLACK)
        SCREEN.blit(user_text, (WIDTH+20, 10))
        algo_text = SMALL_FONT.render(f"Algorithm: {algorithm}", True, BLACK)
        SCREEN.blit(algo_text, (WIDTH+20, 45))
        diff_text = SMALL_FONT.render(f"Difficulty: {difficulty}", True, BLACK)
        SCREEN.blit(diff_text, (WIDTH+20, 80))
        if not game_over:
            if turn == HUMAN:
                turn_txt = MED_FONT.render("Your Turn (Black)", True, BLACK)
            else:
                turn_txt = MED_FONT.render("AI's Turn (White)", True, BLACK)
        else:
            turn_txt = MED_FONT.render("Game Over", True, BLACK)
        SCREEN.blit(turn_txt, (WIDTH+20, 130))
        time_txt = SMALL_FONT.render(f"AI Last Move Time: {ai_last_move_time:.4f}s", True, BLACK)
        SCREEN.blit(time_txt, (WIDTH+20, 170))
        if avg_time is not None:
            avg_txt = SMALL_FONT.render(f"Avg Time: {avg_time:.4f}s", True, BLACK)
            SCREEN.blit(avg_txt, (WIDTH+20, 195))
        moves_txt = SMALL_FONT.render(f"Total Moves: {total_moves}", True, BLACK)
        SCREEN.blit(moves_txt, (WIDTH+20, 220))
        ai_total_txt = SMALL_FONT.render(f"AI Total Time: {ai_total_time:.4f}s", True, BLACK)
        SCREEN.blit(ai_total_txt, (WIDTH+20, 245))
        hc = (board == HUMAN).sum()
        ac = (board == AI).sum()
        cnt_txt = SMALL_FONT.render(f"Black: {hc} | White: {ac}", True, BLACK)
        SCREEN.blit(cnt_txt, (WIDTH+20, 290))
        quit_button = pygame.Rect(WIDTH+20, 340, 120, 40)
        pygame.draw.rect(SCREEN, QUIT_RED, quit_button)
        quit_surf = SMALL_FONT.render("QUIT (Q)", True, WHITE)
        quit_rect = quit_surf.get_rect(center=quit_button.center)
        SCREEN.blit(quit_surf, quit_rect)
        pygame.display.update()
        clock.tick(60)

def place_piece_animated(board, player, row, col,
                         turn, game_over, ai_last_move_time, avg_time, total_moves,
                         ai_total_time=0.0, username="", algorithm="", difficulty="Easy"):
    apply_move(board, row, col, player)
    animate_piece_place(board, row, col, player,
                        turn, game_over, ai_last_move_time, avg_time, total_moves,
                        ai_total_time, username, algorithm, difficulty)

############################################
# AI Hamle Seçimi
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
            best_mv = iterative_deepening_time_move(board, AI, time_limit=1.0)
            return best_mv
        elif algorithm == "Move Ordering":
            best_mv, _ = minimax_move_ordering(board, GAME_MINIMAX_DEPTH, -math.inf, math.inf, True, AI)
            return best_mv
        else:
            best_mv, _ = minimax(board, GAME_MINIMAX_DEPTH, -math.inf, math.inf, True, AI)
            return best_mv


# Q-Learning global değişkeni
q_table = {}

############################################
# Q-Learning Fonksiyonları (Main içinde)
############################################
def get_state_key(board, player):
    return (tuple(board.flatten()), player)

def get_best_q_action(board, player):
    moves = valid_moves(board, player)
    if not moves:
        return None
    st = get_state_key(board, player)
    if st not in q_table:
        return random.choice(moves)
    best_move = None
    best_val = -999999
    for mv in moves:
        val = q_table[st].get(mv, 0.0)
        if val > best_val:
            best_val = val
            best_move = mv
    return best_move if best_move is not None else random.choice(moves)

def q_learning_move(board, player):
    return get_best_q_action(board, player)

def draw_board_training_visual(board, turn, game_over):
    SCREEN.fill(BLACK)
    for r in range(ROWS):
        for c in range(COLUMNS):
            x_pos = c * SQUARESIZE
            y_pos = r * SQUARESIZE
            pygame.draw.rect(SCREEN, DARK_GREEN, (x_pos, y_pos, SQUARESIZE, SQUARESIZE))
            pygame.draw.rect(SCREEN, WHITE, (x_pos, y_pos, SQUARESIZE, SQUARESIZE), 1)
    for r in range(ROWS):
        for c in range(COLUMNS):
            piece = board[r][c]
            if piece != EMPTY:
                color = BLACK if piece == HUMAN else WHITE
                cx = c * SQUARESIZE + SQUARESIZE // 2
                cy = r * SQUARESIZE + SQUARESIZE // 2
                pygame.draw.circle(SCREEN, color, (cx, cy), RADIUS)
    pygame.draw.rect(SCREEN, GREY, (WIDTH, 0, INFO_WIDTH, HEIGHT))
    if not game_over:
        status_txt = MED_FONT.render("Training...", True, BLACK)
    else:
        status_txt = MED_FONT.render("Episode Over", True, BLACK)
    SCREEN.blit(status_txt, (WIDTH+20, 10))
    pygame.display.update()

def train_q_learning_visual_custom(episodes=10, alpha=0.1, gamma=0.95,
                                   epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.99):
    global q_table
    ep = 0
    clock = pygame.time.Clock()

    print("[TRAIN VISUAL] Starting Q-Learning... Press Q to stop early.")

    while ep < episodes:
        board = create_board()
        game_over = False
        turn = random.choice([HUMAN, AI])
        # Q-Learning eğitim döngüsü
        while not game_over:
            draw_board_training_visual(board, turn, game_over)
            pygame.time.wait(30)
            clock.tick(30)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        with open("q_table_othello.pkl", "wb") as f:
                            pickle.dump(q_table, f)
                        print(f"[TRAIN VISUAL] Early exit. {ep} episodes done.")
                        return

            if turn == AI:
                st = get_state_key(board, AI)
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
                    best_val = -999999
                    action = None
                    for mv in moves:
                        val = q_table[st].get(mv, 0.0)
                        if val > best_val:
                            best_val = val
                            action = mv
                row, col = action
                board[row][col] = AI
                apply_move(board, row, col, AI)
                reward = 0.0
                if is_terminal_board(board):
                    w = get_winner(board)
                    if w == "AI":
                        reward = 1.0
                    elif w == "Human":
                        reward = -1.0
                    game_over = True
                new_st = get_state_key(board, HUMAN)
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
                new_q = old_q + alpha * (reward + gamma * max_future - old_q)
                q_table[st][action] = new_q

                if not game_over:
                    turn = HUMAN

            else:  # Turn == HUMAN, rakip = minimax
                moves = valid_moves(board, HUMAN)
                if not moves:
                    turn = AI
                    if not valid_moves(board, AI):
                        game_over = True
                    continue
                best_mv, _ = minimax(board, TRAIN_MINIMAX_DEPTH, -math.inf, math.inf, False, AI)
                if best_mv is None:
                    best_mv = random.choice(moves)
                r, c = best_mv
                board[r][c] = HUMAN
                apply_move(board, r, c, HUMAN)
                if is_terminal_board(board):
                    game_over = True
                else:
                    turn = AI

        ep += 1
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        print(f"[TRAIN VISUAL] Episode {ep}/{episodes} done. Epsilon={epsilon:.3f}")

    with open("q_table_othello.pkl", "wb") as f:
        pickle.dump(q_table, f)
    print("[TRAIN VISUAL] All training done. q_table_othello.pkl saved.")


############################################
# CSV Kaydı
############################################
def write_csv_stats(username, algorithm, difficulty, winner,
                    total_moves, ai_total_time, ai_moves, game_duration):
    filename = f"{username}_stats_othello.csv"
    file_exists = os.path.isfile(filename)
    with open(filename, "a", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "Username","Algorithm","Difficulty","Winner",
            "TotalMoves","AiMoves","AiThinkingTime","GameDuration"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        if not file_exists:
            writer.writeheader()
        row_data = {
            "Username": username,
            "Algorithm": algorithm if algorithm else "Unknown",
            "Difficulty": difficulty,
            "Winner": winner if winner else "Tie",
            "TotalMoves": int(total_moves),
            "AiMoves": int(ai_moves),
            "AiThinkingTime": round(ai_total_time, 4),
            "GameDuration": round(game_duration, 2)
        }
        writer.writerow(row_data)

############################################
# Oyun Döngüsü
############################################
def run_game(username, algorithm, let_me_win, difficulty):
    board = create_board()
    game_over = False
    turn = random.choice([HUMAN, AI])
    minimax_times = []
    ai_total_time = 0.0
    total_moves = 0
    ai_moves = 0
    winner = None
    reason = None
    ai_last_move_time = 0.0
    start_time = time.time()

    while not game_over:
        draw_whole_board(
            board=board,
            turn=turn,
            game_over=game_over,
            ai_last_move_time=ai_last_move_time,
            avg_time=(sum(minimax_times)/len(minimax_times) if minimax_times else None),
            total_moves=total_moves,
            ai_total_time=ai_total_time,
            username=username,
            algorithm=algorithm,
            difficulty=difficulty
        )

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                reason = 'quit'
                return None, minimax_times, total_moves, ai_total_time, board, reason
            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                quit_rect = pygame.Rect(WIDTH+20, 340, 120, 40)
                if quit_rect.collidepoint(mx, my):
                    reason = 'quit'
                    return None, minimax_times, total_moves, ai_total_time, board, reason
                if turn == HUMAN and not game_over:
                    if mx < WIDTH:
                        col = mx // SQUARESIZE
                        row = my // SQUARESIZE
                        vm = valid_moves(board, HUMAN)
                        if (row, col) in vm:
                            place_piece_animated(
                                board, HUMAN, row, col,
                                turn, game_over,
                                ai_last_move_time,
                                (sum(minimax_times)/len(minimax_times) if minimax_times else None),
                                total_moves,
                                ai_total_time,
                                username,
                                algorithm,
                                difficulty
                            )
                            total_moves += 1
                            if is_terminal_board(board):
                                game_over = True
                                winner = get_winner(board)
                            else:
                                if valid_moves(board, AI):
                                    turn = AI
        if not game_over and turn == AI:
            vm = valid_moves(board, AI)
            if not vm:
                turn = HUMAN
            else:
                pygame.time.wait(300)
                start_t = time.time()
                best_mv = ai_move(board, let_me_win=let_me_win, algorithm=algorithm, difficulty=difficulty)
                run_t = time.time() - start_t
                if best_mv is None:
                    turn = HUMAN
                else:
                    minimax_times.append(run_t)
                    ai_total_time += run_t
                    ai_moves += 1
                    total_moves += 1
                    ai_last_move_time = run_t
                    (rr, cc) = best_mv
                    place_piece_animated(
                        board, AI, rr, cc,
                        turn, game_over,
                        ai_last_move_time,
                        (sum(minimax_times)/len(minimax_times) if minimax_times else None),
                        total_moves,
                        ai_total_time,
                        username, algorithm, difficulty
                    )
                    if is_terminal_board(board):
                        game_over = True
                        winner = get_winner(board)
                    else:
                        if valid_moves(board, HUMAN):
                            turn = HUMAN
        if is_terminal_board(board):
            game_over = True
            winner = get_winner(board)

    end_time = time.time()
    game_duration = end_time - start_time

    write_csv_stats(
        username,
        (algorithm if not let_me_win else "Let me win"),
        difficulty,
        winner,
        total_moves,
        ai_total_time,
        ai_moves,
        game_duration
    )

    return winner, minimax_times, total_moves, ai_total_time, board, reason

############################################
# Menü Fonksiyonları (username, algorithm, difficulty, vs.)
############################################
def show_username_screen():
    clock = pygame.time.Clock()
    input_box = pygame.Rect((WIDTH+INFO_WIDTH)//2 -150, HEIGHT//2, 300, 40)
    username = ""
    active_text = False
    play_button = pygame.Rect((WIDTH+INFO_WIDTH)//2 -75, input_box.y+60, 150, 40)
    exit_button = pygame.Rect(20, HEIGHT-60, 100, 40)
    while True:
        clock.tick(30)
        SCREEN.fill((50,50,50))
        title_surf = BIG_FONT.render("Othello / Reversi AI", True, WHITE)
        title_rect = title_surf.get_rect(center=((WIDTH+INFO_WIDTH)//2, HEIGHT//4))
        SCREEN.blit(title_surf, title_rect)
        prompt_surf = SMALL_FONT.render("Enter your username:", True, WHITE)
        SCREEN.blit(prompt_surf, (input_box.x, input_box.y-30))
        pygame.draw.rect(SCREEN, WHITE, input_box, 2)
        name_surf = SMALL_FONT.render(username, True, WHITE)
        SCREEN.blit(name_surf, (input_box.x+5, input_box.y+5))
        pygame.draw.rect(SCREEN, GREY, play_button)
        p_surf = SMALL_FONT.render("Play", True, BLACK)
        p_rect = p_surf.get_rect(center=play_button.center)
        SCREEN.blit(p_surf, p_rect)
        pygame.draw.rect(SCREEN, ORANGE, exit_button)
        e_surf = SMALL_FONT.render("Exit", True, BLACK)
        e_rect = e_surf.get_rect(center=exit_button.center)
        SCREEN.blit(e_surf, e_rect)
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if exit_button.collidepoint(event.pos):
                    pygame.quit()
                    sys.exit()
                if input_box.collidepoint(event.pos):
                    active_text = True
                else:
                    active_text = False
                if play_button.collidepoint(event.pos):
                    if username.strip() == "":
                        username = "player"
                    return username
            elif event.type == pygame.KEYDOWN:
                if active_text:
                    if event.key == pygame.K_RETURN:
                        if username.strip() == "":
                            username = "player"
                        return username
                    elif event.key == pygame.K_BACKSPACE:
                        username = username[:-1]
                    else:
                        username += event.unicode

def show_algorithm_screen():
    clock = pygame.time.Clock()
    button_texts = [
        "A* algorithm",
        "Monte Carlo Tree Search (MCTS)",
        "Min-Max with Alpha-Beta Pruning",
        "Q-Learning",
        "Train Q-Learning (Visual)",
        "Advanced Heuristic Based Search",
        "Negamax Algorithm",
        "Iterative Deepening with Time Constraint",
        "Move Ordering"
    ]
    button_height = 30
    gap = 10
    start_y = 100
    algo_buttons = []
    chosen_algo = "Min-Max with Alpha-Beta Pruning"
    let_me_win = False
    center_x = (WIDTH + INFO_WIDTH) // 2
    button_width = 400
    back_button = pygame.Rect(20, HEIGHT-60, 100, 40)
    for i, txt in enumerate(button_texts):
        rect = pygame.Rect(center_x - button_width//2, start_y + i*(button_height+gap), button_width, button_height)
        algo_buttons.append((rect, txt))
    let_me_win_button = pygame.Rect(center_x - button_width//2, HEIGHT-120, button_width, 40)
    next_button = pygame.Rect(center_x - 75, HEIGHT-60, 150, 40)
    title_y = 50
    while True:
        clock.tick(30)
        SCREEN.fill((50,50,50))
        prompt_surf = MED_FONT.render("Choose algorithm:", True, WHITE)
        prompt_rect = prompt_surf.get_rect(center=(center_x, title_y))
        SCREEN.blit(prompt_surf, prompt_rect)
        for (rct, text) in algo_buttons:
            col = ORANGE if text == chosen_algo else GREY
            pygame.draw.rect(SCREEN, col, rct)
            t_surf = SMALL_FONT.render(text, True, BLACK)
            t_rect = t_surf.get_rect(center=rct.center)
            SCREEN.blit(t_surf, t_rect)
        lw_col = ORANGE if let_me_win else GREY
        pygame.draw.rect(SCREEN, lw_col, let_me_win_button)
        lw_txt = SMALL_FONT.render("Let me win", True, BLACK)
        lw_rect = lw_txt.get_rect(center=let_me_win_button.center)
        SCREEN.blit(lw_txt, lw_rect)
        pygame.draw.rect(SCREEN, GREY, next_button)
        n_surf = SMALL_FONT.render("Next", True, BLACK)
        n_rect = n_surf.get_rect(center=next_button.center)
        SCREEN.blit(n_surf, n_rect)
        pygame.draw.rect(SCREEN, ORANGE, back_button)
        b_surf = SMALL_FONT.render("Back", True, BLACK)
        b_rect = b_surf.get_rect(center=back_button.center)
        SCREEN.blit(b_surf, b_rect)
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                for (rct, txt_name) in algo_buttons:
                    if rct.collidepoint(event.pos):
                        chosen_algo = txt_name
                if let_me_win_button.collidepoint(event.pos):
                    let_me_win = not let_me_win
                if next_button.collidepoint(event.pos):
                    return chosen_algo, let_me_win, None
                if back_button.collidepoint(event.pos):
                    return None, None, 'back'

def show_episodes_screen_for_visual():
    clock = pygame.time.Clock()
    input_box = pygame.Rect((WIDTH+INFO_WIDTH)//2 -150, HEIGHT//2, 300, 40)
    user_input = ""
    active_text = False
    train_button = pygame.Rect((WIDTH+INFO_WIDTH)//2 -75, input_box.y+60, 150, 40)
    back_button = pygame.Rect(20, HEIGHT-60, 100, 40)
    while True:
        clock.tick(30)
        SCREEN.fill((50,50,50))
        title_surf = MED_FONT.render("Enter number of episodes:", True, WHITE)
        title_rect = title_surf.get_rect(center=((WIDTH+INFO_WIDTH)//2, HEIGHT//4))
        SCREEN.blit(title_surf, title_rect)
        pygame.draw.rect(SCREEN, WHITE, input_box, 2)
        ep_surf = SMALL_FONT.render(user_input, True, WHITE)
        SCREEN.blit(ep_surf, (input_box.x+5, input_box.y+5))
        pygame.draw.rect(SCREEN, GREY, train_button)
        t_surf = SMALL_FONT.render("Train", True, BLACK)
        t_rect = t_surf.get_rect(center=train_button.center)
        SCREEN.blit(t_surf, t_rect)
        pygame.draw.rect(SCREEN, ORANGE, back_button)
        b_surf = SMALL_FONT.render("Back", True, BLACK)
        b_rect = b_surf.get_rect(center=back_button.center)
        SCREEN.blit(b_surf, b_rect)
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if input_box.collidepoint(event.pos):
                    active_text = True
                else:
                    active_text = False
                if train_button.collidepoint(event.pos):
                    if user_input.strip() == "":
                        return 10, None
                    else:
                        return int(user_input), None
                if back_button.collidepoint(event.pos):
                    return None, 'back'
            elif event.type == pygame.KEYDOWN:
                if active_text:
                    if event.key == pygame.K_RETURN:
                        if user_input.strip() == "":
                            return 10, None
                        else:
                            return int(user_input), None
                    elif event.key == pygame.K_BACKSPACE:
                        user_input = user_input[:-1]
                    elif event.unicode.isdigit():
                        user_input += event.unicode

def show_difficulty_screen():
    clock = pygame.time.Clock()
    difficulties = ["Easy", "Medium", "Hard", "Extreme"]
    button_height = 50
    gap = 20
    start_y = HEIGHT//2 - 120
    diff_buttons = []
    chosen_diff = "Easy"
    back_button = pygame.Rect(20, HEIGHT-60, 100, 40)
    for i, text in enumerate(difficulties):
        rect = pygame.Rect((WIDTH+INFO_WIDTH)//2 -150, start_y + i*(button_height+gap), 300, button_height)
        diff_buttons.append((rect, text))
    start_button = pygame.Rect((WIDTH+INFO_WIDTH)//2 -75, start_y + 4*(button_height+gap) + 50, 150, 40)
    while True:
        clock.tick(30)
        SCREEN.fill((50,50,50))
        prompt_surf = MED_FONT.render("Choose difficulty:", True, WHITE)
        prompt_rect = prompt_surf.get_rect(center=((WIDTH+INFO_WIDTH)//2, HEIGHT//4))
        SCREEN.blit(prompt_surf, prompt_rect)
        for (rct, dt) in diff_buttons:
            col = ORANGE if dt == chosen_diff else GREY
            pygame.draw.rect(SCREEN, col, rct)
            dt_surf = SMALL_FONT.render(dt, True, BLACK)
            dt_rect = dt_surf.get_rect(center=rct.center)
            SCREEN.blit(dt_surf, dt_rect)
        pygame.draw.rect(SCREEN, GREY, start_button)
        st_surf = SMALL_FONT.render("Start Game", True, BLACK)
        st_rect = st_surf.get_rect(center=start_button.center)
        SCREEN.blit(st_surf, st_rect)
        pygame.draw.rect(SCREEN, ORANGE, back_button)
        b_surf = SMALL_FONT.render("Back", True, BLACK)
        b_rect = b_surf.get_rect(center=back_button.center)
        SCREEN.blit(b_surf, b_rect)
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                for (rct, dt_) in diff_buttons:
                    if rct.collidepoint(event.pos):
                        chosen_diff = dt_
                if start_button.collidepoint(event.pos):
                    return chosen_diff, None
                if back_button.collidepoint(event.pos):
                    return None, 'back'

############################################
# Game Over & End Screen
############################################
def show_game_over_screen():
    overlay = pygame.Surface((WIDTH+INFO_WIDTH, HEIGHT))
    overlay.fill((0,0,0))
    overlay.set_alpha(180)
    txt = BIG_FONT.render("Game Over", True, WHITE)
    txt_rect = txt.get_rect(center=((WIDTH+INFO_WIDTH)//2, HEIGHT//2))
    SCREEN.blit(overlay, (0,0))
    SCREEN.blit(txt, txt_rect)
    pygame.display.update()
    pygame.time.wait(1500)

def show_end_screen(winner, total_moves, ai_total_time, board):
    SCREEN.fill((50,50,50))
    lines = []
    hc = (board == HUMAN).sum()
    ac = (board == AI).sum()
    if winner == "AI":
        lines = [
            "AI (White) wins!",
            f"Score: (Black) {hc} - (White) {ac}",
            f"AI Thinking Time: {ai_total_time:.4f}s",
            f"Total Moves: {total_moves}",
            "Play again?"
        ]
        color = WHITE
    elif winner == "Human":
        lines = [
            "You (Black) win!",
            f"Score: (Black) {hc} - (White) {ac}",
            f"Total Moves: {total_moves}",
            "Play again?"
        ]
        color = BLACK
    else:
        lines = [
            "It's a tie!",
            f"Score: (Black) {hc} - (White) {ac}",
            f"Total Moves: {total_moves}",
            "Play again?"
        ]
        color = GREY
    line_height = 50
    total_text_height = len(lines) * line_height
    start_y = (HEIGHT - total_text_height) // 2
    center_x = (WIDTH+INFO_WIDTH)//2
    for i, line in enumerate(lines):
        surf = BIG_FONT.render(line, True, color)
        rect = surf.get_rect(center=(center_x, start_y + i*line_height))
        SCREEN.blit(surf, rect)
    yes_button = pygame.Rect(center_x-120, start_y + len(lines)*line_height + 30, 80, 40)
    no_button  = pygame.Rect(center_x+40, start_y + len(lines)*line_height + 30, 80, 40)
    pygame.draw.rect(SCREEN, GREY, yes_button)
    pygame.draw.rect(SCREEN, GREY, no_button)
    yes_text = SMALL_FONT.render("YES", True, BLACK)
    no_text = SMALL_FONT.render("NO", True, BLACK)
    yes_rect = yes_text.get_rect(center=yes_button.center)
    no_rect = no_text.get_rect(center=no_button.center)
    SCREEN.blit(yes_text, yes_rect)
    SCREEN.blit(no_text, no_rect)
    pygame.display.update()
    clock = pygame.time.Clock()
    while True:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if yes_button.collidepoint(event.pos):
                    return True
                elif no_button.collidepoint(event.pos):
                    return False

############################################
# main Fonksiyonu
############################################
def main():
    global q_table
    if os.path.exists("q_table_othello.pkl"):
        with open("q_table_othello.pkl", "rb") as f:
            q_table = pickle.load(f)
        print("[INFO] q_table_othello.pkl loaded.")
    else:
        print("[INFO] No q_table_othello.pkl found. Starting fresh...")
        q_table = {}

    username = show_username_screen()

    while True:
        while True:
            algorithm, let_me_win, reason_algo = show_algorithm_screen()
            if reason_algo == 'back':
                username = show_username_screen()
            else:
                break

        if algorithm == "Train Q-Learning (Visual)":
            while True:
                ep_count, reason_ep = show_episodes_screen_for_visual()
                if reason_ep == 'back':
                    break
                else:
                    print(f"[INFO] Visual Q-Learning train for {ep_count} episodes.")
                    train_q_learning_visual_custom(episodes=ep_count)
                    break
            continue

        while True:
            difficulty, reason_diff = show_difficulty_screen()
            if reason_diff == 'back':
                break
            if difficulty is not None:
                winner, minimax_times, total_moves, ai_total_time, final_board, reason_game = run_game(
                    username, algorithm, let_me_win, difficulty
                )
                if reason_game == 'quit':
                    print("[INFO] QUIT in game -> returning to algorithm screen.")
                    break
                show_game_over_screen()
                play_again = show_end_screen(winner, total_moves, ai_total_time, final_board)
                if not play_again:
                    pygame.quit()
                    sys.exit()
                break
        # Döngü algoritma ekranına geri dönüyor

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
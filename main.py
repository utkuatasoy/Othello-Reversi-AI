# main.py

import pygame
import sys
import time
import csv
import os

from .constants import (
    SCREEN, WIDTH, HEIGHT, INFO_WIDTH,
    HUMAN, AI,
    SMALL_FONT, GAME_MINIMAX_DEPTH
)
from .board import (
    create_board, valid_moves, is_terminal_board, get_winner,
    get_opponent
)
from .draw import (
    draw_whole_board, place_piece_animated,
    show_game_over_screen, show_end_screen
)
from .ai_move import (
    ai_move, q_table, train_q_learning_visual_custom
)
from .menu import (
    show_username_screen, show_algorithm_screen,
    show_episodes_screen_for_visual, show_difficulty_screen
)


def write_csv_stats(username, algorithm, difficulty, winner, total_moves, ai_total_time, ai_moves, game_duration):
    """
    Oyun bitişinde CSV'ye ekle.
    """
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
            "AiThinkingTime": round(ai_total_time,4),
            "GameDuration": round(game_duration,2)
        }
        writer.writerow(row_data)

def run_game(username, algorithm, let_me_win, difficulty):
    board = create_board()
    game_over = False
    import random
    turn = random.choice([HUMAN, AI])
    minimax_times = []
    ai_total_time = 0.0
    total_moves   = 0
    ai_moves      = 0
    winner        = None
    reason        = None

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
                quit_rect = pygame.Rect(WIDTH+20, 340, 120,40)
                if quit_rect.collidepoint(mx,my):
                    reason = 'quit'
                    return None, minimax_times, total_moves, ai_total_time, board, reason

                if turn == HUMAN and not game_over:
                    if mx < WIDTH:  # Tahta içine tıklama
                        col = mx // 80
                        row = my // 80
                        vm = valid_moves(board, HUMAN)
                        if (row,col) in vm:
                            place_piece_animated(
                                board, HUMAN, row, col, turn, game_over,
                                ai_last_move_time,
                                (sum(minimax_times)/len(minimax_times) if minimax_times else None),
                                total_moves,
                                ai_total_time,
                                username, algorithm, difficulty,
                                apply_move_func=None  # apply_move fonk. draw.py içinde parametrelenmiş
                            )
                            total_moves += 1
                            if is_terminal_board(board):
                                game_over = True
                                winner = get_winner(board)
                            else:
                                if valid_moves(board, AI):
                                    turn = AI
                                else:
                                    pass  # AI pas

        # AI Hamlesi
        if not game_over and turn == AI:
            vm = valid_moves(board, AI)
            if not vm:
                turn = HUMAN
            else:
                pygame.time.wait(300)  # ufak bekleme
                start_t = time.time()
                best_mv = ai_move(board, let_me_win=let_me_win, algorithm=algorithm, difficulty=difficulty)
                run_t   = time.time() - start_t

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
                        board, AI, rr, cc, turn, game_over,
                        ai_last_move_time,
                        (sum(minimax_times)/len(minimax_times) if minimax_times else None),
                        total_moves,
                        ai_total_time,
                        username, algorithm, difficulty,
                        apply_move_func=None
                    )

                    if is_terminal_board(board):
                        game_over = True
                        winner = get_winner(board)
                    else:
                        if not valid_moves(board, HUMAN):
                            pass
                        else:
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

def main():
    # Q-table yükle
    if os.path.exists("q_table_othello.pkl"):
        import pickle
        with open("q_table_othello.pkl","rb") as f:
            loaded_q = pickle.load(f)
        from .ai_move import q_table
        q_table.update(loaded_q)
        print("[INFO] q_table_othello.pkl loaded.")
    else:
        print("[INFO] No q_table_othello.pkl found. Starting fresh...")

    username = show_username_screen()

    while True:
        # Algoritma seçimi
        while True:
            algorithm, let_me_win, reason_algo = show_algorithm_screen()
            if reason_algo == 'back':
                username = show_username_screen()
            else:
                break

        # Train Q-Learning (Visual) seçildiyse
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

        # Zorluk seçimi
        while True:
            difficulty, reason_diff = show_difficulty_screen()
            if reason_diff == 'back':
                break
            if difficulty is not None:
                winner, minimax_times, total_moves, ai_total_time, final_board, reason_game = run_game(
                    username, algorithm, let_me_win, difficulty
                )
                if reason_game == 'quit':
                    print("[INFO] QUIT in game -> returning to algo screen.")
                    break

                # Normal bitiş
                show_game_over_screen()
                play_again = show_end_screen(winner, total_moves, ai_total_time, final_board)
                if not play_again:
                    pygame.quit()
                    sys.exit()
                # Evet -> yine zorluk menüsüne dön
                break

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()

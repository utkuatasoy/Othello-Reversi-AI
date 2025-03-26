# draw.py

import pygame
import time
import random
import numpy as np
from .constants import (
    SCREEN, WIDTH, HEIGHT, INFO_WIDTH,
    SQUARESIZE, RADIUS, EMPTY, HUMAN, AI,
    BLACK, WHITE, GREY, DARK_GREEN, QUIT_RED,
    SMALL_FONT, MED_FONT, BIG_FONT
)
from .board import valid_moves

def draw_whole_board(
    board, turn, game_over, ai_last_move_time, avg_time, total_moves,
    ai_total_time=0.0, username="", algorithm="", difficulty="Easy"
):
    """
    Ekrandaki tahtayı ve sağ paneli çizer.
    """
    SCREEN.fill(BLACK)

    # Tahta kareleri (koyu yeşil + beyaz çerçeve)
    for r in range(board.shape[0]):
        for c in range(board.shape[1]):
            x_pos = c * SQUARESIZE
            y_pos = r * SQUARESIZE
            pygame.draw.rect(SCREEN, DARK_GREEN, (x_pos, y_pos, SQUARESIZE, SQUARESIZE))
            pygame.draw.rect(SCREEN, WHITE, (x_pos, y_pos, SQUARESIZE, SQUARESIZE), 1)

    # Tahtadaki taşlar
    for r in range(board.shape[0]):
        for c in range(board.shape[1]):
            piece = board[r][c]
            if piece != EMPTY:
                color = BLACK if piece == HUMAN else WHITE
                cx = c * SQUARESIZE + SQUARESIZE // 2
                cy = r * SQUARESIZE + SQUARESIZE // 2
                pygame.draw.circle(SCREEN, color, (cx, cy), RADIUS)

    # İnsan sırası ise geçerli hamleleri küçük daireyle göster
    if not game_over and turn == HUMAN:
        moves = valid_moves(board, HUMAN)
        for (rr, cc) in moves:
            cx = cc * SQUARESIZE + SQUARESIZE // 2
            cy = rr * SQUARESIZE + SQUARESIZE // 2
            pygame.draw.circle(SCREEN, (0,200,0), (cx, cy), 8)

    # Sağ panel
    pygame.draw.rect(SCREEN, GREY, (WIDTH, 0, INFO_WIDTH, HEIGHT))

    # Panel yazıları
    user_text = SMALL_FONT.render(f"Player: {username}", True, BLACK)
    SCREEN.blit(user_text, (WIDTH + 20, 10))

    algo_text = SMALL_FONT.render(f"Algorithm: {algorithm}", True, BLACK)
    SCREEN.blit(algo_text, (WIDTH + 20, 45))

    diff_text = SMALL_FONT.render(f"Difficulty: {difficulty}", True, BLACK)
    SCREEN.blit(diff_text, (WIDTH + 20, 80))

    if not game_over:
        if turn == HUMAN:
            turn_txt = MED_FONT.render("Your Turn (Black)", True, BLACK)
        else:
            turn_txt = MED_FONT.render("AI's Turn (White)", True, BLACK)
    else:
        turn_txt = MED_FONT.render("Game Over", True, BLACK)
    SCREEN.blit(turn_txt, (WIDTH + 20, 130))

    # AI Last Move Time
    time_txt = SMALL_FONT.render(f"AI Last Move Time: {ai_last_move_time:.4f}s", True, BLACK)
    SCREEN.blit(time_txt, (WIDTH + 20, 170))

    # Avg time
    if avg_time is not None:
        avg_txt = SMALL_FONT.render(f"Avg Time: {avg_time:.4f}s", True, BLACK)
        SCREEN.blit(avg_txt, (WIDTH + 20, 195))

    moves_txt = SMALL_FONT.render(f"Total Moves: {total_moves}", True, BLACK)
    SCREEN.blit(moves_txt, (WIDTH + 20, 220))

    ai_total_txt = SMALL_FONT.render(f"AI Total Time: {ai_total_time:.4f}s", True, BLACK)
    SCREEN.blit(ai_total_txt, (WIDTH + 20, 245))

    # Siyah-beyaz taş sayısı
    human_count = np.count_nonzero(board == HUMAN)
    ai_count    = np.count_nonzero(board == AI)
    cnt_txt = SMALL_FONT.render(f"Black: {human_count} | White: {ai_count}", True, BLACK)
    SCREEN.blit(cnt_txt, (WIDTH + 20, 290))

    # Quit butonu
    quit_button = pygame.Rect(WIDTH + 20, 340, 120, 40)
    pygame.draw.rect(SCREEN, QUIT_RED, quit_button)
    quit_surf = SMALL_FONT.render("QUIT (Q)", True, WHITE)
    quit_rect = quit_surf.get_rect(center=quit_button.center)
    SCREEN.blit(quit_surf, quit_rect)

    pygame.display.update()


def animate_piece_place(
    board, row, col, player, turn, game_over,
    ai_last_move_time, avg_time, total_moves,
    ai_total_time=0.0, username="", algorithm="", difficulty="Easy"
):
    """
    Basit bir animasyon: Taş yavaşça büyüyor.
    """
    clock = pygame.time.Clock()
    color = BLACK if player == HUMAN else WHITE
    steps = 12

    for step in range(1, steps + 1):
        SCREEN.fill(BLACK)

        # Tahta kareleri
        for rr in range(board.shape[0]):
            for cc in range(board.shape[1]):
                x_pos = cc * SQUARESIZE
                y_pos = rr * SQUARESIZE
                pygame.draw.rect(SCREEN, DARK_GREEN, (x_pos, y_pos, SQUARESIZE, SQUARESIZE))
                pygame.draw.rect(SCREEN, WHITE, (x_pos, y_pos, SQUARESIZE, SQUARESIZE), 1)

        # Mevcut taşlar
        for rr in range(board.shape[0]):
            for cc in range(board.shape[1]):
                piece = board[rr][cc]
                if piece != EMPTY:
                    clr = BLACK if piece == HUMAN else WHITE
                    cx = cc * SQUARESIZE + SQUARESIZE // 2
                    cy = rr * SQUARESIZE + SQUARESIZE // 2
                    pygame.draw.circle(SCREEN, clr, (cx, cy), RADIUS)

        # Yeni taş
        cx_new = col * SQUARESIZE + SQUARESIZE // 2
        cy_new = row * SQUARESIZE + SQUARESIZE // 2
        current_radius = int(RADIUS * (step / steps))
        pygame.draw.circle(SCREEN, color, (cx_new, cy_new), current_radius)

        # Sağ panel
        pygame.draw.rect(SCREEN, GREY, (WIDTH, 0, INFO_WIDTH, HEIGHT))

        user_text = SMALL_FONT.render(f"Player: {username}", True, BLACK)
        SCREEN.blit(user_text, (WIDTH + 20, 10))

        algo_text = SMALL_FONT.render(f"Algorithm: {algorithm}", True, BLACK)
        SCREEN.blit(algo_text, (WIDTH + 20, 45))

        diff_text = SMALL_FONT.render(f"Difficulty: {difficulty}", True, BLACK)
        SCREEN.blit(diff_text, (WIDTH + 20, 80))

        if not game_over:
            if turn == HUMAN:
                turn_txt = MED_FONT.render("Your Turn (Black)", True, BLACK)
            else:
                turn_txt = MED_FONT.render("AI's Turn (White)", True, BLACK)
        else:
            turn_txt = MED_FONT.render("Game Over", True, BLACK)
        SCREEN.blit(turn_txt, (WIDTH + 20, 130))

        time_txt = SMALL_FONT.render(f"AI Last Move Time: {ai_last_move_time:.4f}s", True, BLACK)
        SCREEN.blit(time_txt, (WIDTH + 20, 170))

        if avg_time is not None:
            avg_txt = SMALL_FONT.render(f"Avg Time: {avg_time:.4f}s", True, BLACK)
            SCREEN.blit(avg_txt, (WIDTH + 20, 195))

        moves_txt = SMALL_FONT.render(f"Total Moves: {total_moves}", True, BLACK)
        SCREEN.blit(moves_txt, (WIDTH + 20, 220))

        ai_total_txt = SMALL_FONT.render(f"AI Total Time: {ai_total_time:.4f}s", True, BLACK)
        SCREEN.blit(ai_total_txt, (WIDTH + 20, 245))

        hc = np.count_nonzero(board == HUMAN)
        ac = np.count_nonzero(board == AI)
        cnt_txt = SMALL_FONT.render(f"Black: {hc} | White: {ac}", True, BLACK)
        SCREEN.blit(cnt_txt, (WIDTH + 20, 290))

        # Quit butonu
        quit_button = pygame.Rect(WIDTH + 20, 340, 120, 40)
        pygame.draw.rect(SCREEN, QUIT_RED, quit_button)
        quit_surf = SMALL_FONT.render("QUIT (Q)", True, WHITE)
        quit_rect = quit_surf.get_rect(center=quit_button.center)
        SCREEN.blit(quit_surf, quit_rect)

        pygame.display.update()
        clock.tick(60)

def draw_board_training_visual(board, turn, game_over):
    """
    Q-Learning eğitimi sırasında her hamlede tahtayı kısaca göstermek için 
    kullanılan sade bir görselleştirme.
    """
    SCREEN.fill(BLACK)

    # Tahta kareleri
    for r in range(board.shape[0]):
        for c in range(board.shape[1]):
            x_pos = c * SQUARESIZE
            y_pos = r * SQUARESIZE
            pygame.draw.rect(SCREEN, DARK_GREEN, (x_pos, y_pos, SQUARESIZE, SQUARESIZE))
            pygame.draw.rect(SCREEN, WHITE, (x_pos, y_pos, SQUARESIZE, SQUARESIZE), 1)

    # Taşlar
    for r in range(board.shape[0]):
        for c in range(board.shape[1]):
            piece = board[r][c]
            if piece != 0:
                color = BLACK if piece == HUMAN else WHITE
                cx = c * SQUARESIZE + SQUARESIZE // 2
                cy = r * SQUARESIZE + SQUARESIZE // 2
                pygame.draw.circle(SCREEN, color, (cx, cy), RADIUS)

    # Sağ tarafa basit metin
    import math
    pygame.draw.rect(SCREEN, GREY, (WIDTH, 0, INFO_WIDTH, HEIGHT))
    if not game_over:
        status_txt = MED_FONT.render("Training...", True, BLACK)
    else:
        status_txt = MED_FONT.render("Episode Over", True, BLACK)
    SCREEN.blit(status_txt, (WIDTH + 20, 10))

    pygame.display.update()

def place_piece_animated(
    board, player, row, col, turn, game_over,
    ai_last_move_time, avg_time, total_moves,
    ai_total_time=0.0, username="", algorithm="", difficulty="Easy",
    apply_move_func=None
):
    """
    apply_move + animasyon
    """
    # move uygulama
    if apply_move_func:
        apply_move_func(board, row, col, player)

    # animasyon
    animate_piece_place(
        board, row, col, player,
        turn, game_over,
        ai_last_move_time, avg_time, total_moves,
        ai_total_time, username, algorithm, difficulty
    )

def show_game_over_screen():
    """
    Oyun biter bitmez ekrana yarı saydam bir "Game Over" yazısı
    """
    overlay = pygame.Surface((WIDTH + INFO_WIDTH, HEIGHT))
    overlay.fill((0,0,0))
    overlay.set_alpha(180)
    txt = BIG_FONT.render("Game Over", True, WHITE)
    txt_rect = txt.get_rect(center=((WIDTH + INFO_WIDTH)//2, HEIGHT//2))
    SCREEN.blit(overlay, (0,0))
    SCREEN.blit(txt, txt_rect)
    pygame.display.update()
    pygame.time.wait(1500)

def show_end_screen(winner, total_moves, ai_total_time, board):
    """
    Winner'a göre ekrana yazı. "Play again?" -> YES/NO
    """
    SCREEN.fill((50,50,50))
    import numpy as np
    hc = np.count_nonzero(board == HUMAN)
    ac = np.count_nonzero(board == AI)

    lines = []
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
        color = (0,0,0)
    else:
        lines = [
            "It's a tie!",
            f"Score: (Black) {hc} - (White) {ac}",
            f"Total Moves: {total_moves}",
            "Play again?"
        ]
        color = GREY

    line_height = 50
    start_y = (HEIGHT - len(lines)*line_height)//2
    center_x = (WIDTH + INFO_WIDTH)//2

    import pygame
    for i, line in enumerate(lines):
        surf = BIG_FONT.render(line, True, color)
        rect = surf.get_rect(center=(center_x, start_y + i * line_height))
        SCREEN.blit(surf, rect)

    # YES/NO butonları
    yes_button = pygame.Rect(center_x - 120, start_y + len(lines)*line_height + 30, 80, 40)
    no_button  = pygame.Rect(center_x + 40,  start_y + len(lines)*line_height + 30, 80, 40)

    pygame.draw.rect(SCREEN, GREY, yes_button)
    pygame.draw.rect(SCREEN, GREY, no_button)

    yes_text = SMALL_FONT.render("YES", True, BLACK)
    no_text  = SMALL_FONT.render("NO",  True, BLACK)

    yes_rect = yes_text.get_rect(center=yes_button.center)
    no_rect  = no_text.get_rect(center=no_button.center)

    SCREEN.blit(yes_text, yes_rect)
    SCREEN.blit(no_text, no_rect)

    pygame.display.update()
    clock = pygame.time.Clock()
    while True:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                import sys
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if yes_button.collidepoint(event.pos):
                    return True
                elif no_button.collidepoint(event.pos):
                    return False

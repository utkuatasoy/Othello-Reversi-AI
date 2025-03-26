# menu.py

import pygame
import sys
from .constants import (
    SCREEN, WIDTH, HEIGHT, INFO_WIDTH,
    MED_FONT, SMALL_FONT, ORANGE, GREY
)

def show_username_screen():
    """
    Kullanıcı adı giriş ekranı.
    """
    clock = pygame.time.Clock()
    input_box = pygame.Rect((WIDTH+INFO_WIDTH)//2 - 150, HEIGHT//2, 300, 40)
    username = ""
    active_text = False
    play_button = pygame.Rect((WIDTH+INFO_WIDTH)//2 - 75, input_box.y+60, 150, 40)
    exit_button = pygame.Rect(20, HEIGHT - 60, 100, 40)  # 'Exit'

    while True:
        clock.tick(30)
        SCREEN.fill((50,50,50))

        import pygame
        title_surf = pygame.font.SysFont("Arial", 48, bold=True).render("Othello / Reversi AI", True, (255,255,255))
        title_rect = title_surf.get_rect(center=((WIDTH+INFO_WIDTH)//2, HEIGHT//4))
        SCREEN.blit(title_surf, title_rect)

        prompt_surf = SMALL_FONT.render("Enter your username:", True, (255,255,255))
        SCREEN.blit(prompt_surf, (input_box.x, input_box.y-30))

        pygame.draw.rect(SCREEN, (255,255,255), input_box, 2)
        name_surf = SMALL_FONT.render(username, True, (255,255,255))
        SCREEN.blit(name_surf, (input_box.x+5, input_box.y+5))

        pygame.draw.rect(SCREEN, GREY, play_button)
        p_surf = SMALL_FONT.render("Play", True, (0,0,0))
        p_rect = p_surf.get_rect(center=play_button.center)
        SCREEN.blit(p_surf, p_rect)

        # Exit butonu
        pygame.draw.rect(SCREEN, ORANGE, exit_button)
        e_surf = SMALL_FONT.render("Exit", True, (0,0,0))
        e_rect = e_surf.get_rect(center=exit_button.center)
        SCREEN.blit(e_surf, e_rect)

        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Exit
                if exit_button.collidepoint(event.pos):
                    pygame.quit()
                    sys.exit()

                # Input box
                if input_box.collidepoint(event.pos):
                    active_text = True
                else:
                    active_text = False

                # Play
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
    """
    Algoritma seçimi menüsü.
    """
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

    # Back butonu
    back_button = pygame.Rect(20, HEIGHT - 60, 100, 40)

    # Algoritma butonları
    for i, txt in enumerate(button_texts):
        rect = pygame.Rect(center_x - button_width // 2, start_y + i*(button_height+gap), button_width, button_height)
        algo_buttons.append((rect, txt))

    # "Let me win" butonu
    let_me_win_button = pygame.Rect(center_x - button_width // 2, HEIGHT - 120, button_width, 40)
    next_button = pygame.Rect(center_x - 75, HEIGHT - 60, 150, 40)

    title_y = 50

    while True:
        clock.tick(30)
        SCREEN.fill((50,50,50))

        prompt_surf = MED_FONT.render("Choose algorithm:", True, (255,255,255))
        prompt_rect = prompt_surf.get_rect(center=(center_x, title_y))
        SCREEN.blit(prompt_surf, prompt_rect)

        for (rct, text) in algo_buttons:
            col = ORANGE if text == chosen_algo else GREY
            pygame.draw.rect(SCREEN, col, rct)
            t_surf = SMALL_FONT.render(text, True, (0,0,0))
            t_rect = t_surf.get_rect(center=rct.center)
            SCREEN.blit(t_surf, t_rect)

        lw_col = ORANGE if let_me_win else GREY
        pygame.draw.rect(SCREEN, lw_col, let_me_win_button)
        lw_txt = SMALL_FONT.render("Let me win", True, (0,0,0))
        lw_rect = lw_txt.get_rect(center=let_me_win_button.center)
        SCREEN.blit(lw_txt, lw_rect)

        pygame.draw.rect(SCREEN, GREY, next_button)
        n_surf = SMALL_FONT.render("Next", True, (0,0,0))
        n_rect = n_surf.get_rect(center=next_button.center)
        SCREEN.blit(n_surf, n_rect)

        # Back
        pygame.draw.rect(SCREEN, ORANGE, back_button)
        b_surf = SMALL_FONT.render("Back", True, (0,0,0))
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
    """
    Q-Learning train -> episode sayısı
    """
    clock = pygame.time.Clock()
    input_box = pygame.Rect((WIDTH+INFO_WIDTH)//2 -150, HEIGHT//2, 300,40)
    user_input = ""
    active_text = False
    train_button = pygame.Rect((WIDTH+INFO_WIDTH)//2 -75, input_box.y+60, 150, 40)
    back_button  = pygame.Rect(20, HEIGHT-60, 100, 40)

    while True:
        clock.tick(30)
        SCREEN.fill((50,50,50))

        import pygame
        title_surf = MED_FONT.render("Enter number of episodes:", True, (255,255,255))
        title_rect = title_surf.get_rect(center=((WIDTH+INFO_WIDTH)//2, HEIGHT//4))
        SCREEN.blit(title_surf, title_rect)

        pygame.draw.rect(SCREEN, (255,255,255), input_box, 2)
        ep_surf = SMALL_FONT.render(user_input, True, (255,255,255))
        SCREEN.blit(ep_surf, (input_box.x+5, input_box.y+5))

        pygame.draw.rect(SCREEN, GREY, train_button)
        t_surf = SMALL_FONT.render("Train", True, (0,0,0))
        t_rect = t_surf.get_rect(center=train_button.center)
        SCREEN.blit(t_surf, t_rect)

        # Back
        pygame.draw.rect(SCREEN, ORANGE, back_button)
        b_surf = SMALL_FONT.render("Back", True, (0,0,0))
        b_rect = b_surf.get_rect(center= back_button.center)
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
    """
    Zorluk seçimi ekranı
    """
    clock = pygame.time.Clock()
    difficulties = ["Easy", "Medium", "Hard", "Extreme"]
    button_height = 50
    gap = 20
    start_y = (HEIGHT // 2) - 120
    diff_buttons = []
    chosen_diff = "Easy"

    back_button = pygame.Rect(20, HEIGHT - 60, 100, 40)

    for i, text in enumerate(difficulties):
        rect = pygame.Rect((WIDTH+INFO_WIDTH)//2 -150, start_y + i*(button_height+gap), 300, button_height)
        diff_buttons.append((rect, text))

    start_button = pygame.Rect((WIDTH+INFO_WIDTH)//2 -75, start_y + 4*(button_height+gap) + 50, 150, 40)

    while True:
        clock.tick(30)
        SCREEN.fill((50,50,50))

        prompt_surf = MED_FONT.render("Choose difficulty:", True, (255,255,255))
        prompt_rect = prompt_surf.get_rect(center=((WIDTH+INFO_WIDTH)//2, HEIGHT//4))
        SCREEN.blit(prompt_surf, prompt_rect)

        for (rct, dt) in diff_buttons:
            import pygame
            col = (255,140,0) if dt == chosen_diff else (200,200,200)
            pygame.draw.rect(SCREEN, col, rct)
            dt_surf = SMALL_FONT.render(dt, True, (0,0,0))
            dt_rect = dt_surf.get_rect(center=rct.center)
            SCREEN.blit(dt_surf, dt_rect)

        pygame.draw.rect(SCREEN, (200,200,200), start_button)
        st_surf = SMALL_FONT.render("Start Game", True, (0,0,0))
        st_rect = st_surf.get_rect(center=start_button.center)
        SCREEN.blit(st_surf, st_rect)

        # Back
        pygame.draw.rect(SCREEN, ORANGE, back_button)
        b_surf = SMALL_FONT.render("Back", True, (0,0,0))
        b_rect = b_surf.get_rect(center= back_button.center)
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

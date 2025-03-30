import pygame
import sys
import math
import random
import time
import csv
import os
import numpy as np
import pickle

############################################
# Q-Table
############################################
q_table = {}

############################################
# Othello / Reversi Parametreler
############################################
ROWS = 8
COLUMNS = 8
EMPTY = 0
HUMAN = 1   # Siyah (örnek)
AI = 2      # Beyaz (örnek)

SQUARESIZE = 80
RADIUS = SQUARESIZE // 2 - 4

WIDTH = COLUMNS * SQUARESIZE   # 640
HEIGHT = ROWS * SQUARESIZE     # 640
INFO_WIDTH = 360
SCREEN_SIZE = (WIDTH + INFO_WIDTH, HEIGHT)

# Renkler
DARK_GREEN  = (0,120,0)        # Tahta zemin
BLACK       = (0,0,0)
WHITE       = (255,255,255)
GREY        = (200,200,200)
ORANGE      = (255,140,0)
RED         = (255,0,0)
BLUE        = (30,110,250)
YELLOW      = (255,255,0)
QUIT_RED    = (200,0,0)        # Quit butonu rengi

TRAIN_MINIMAX_DEPTH = 2
GAME_MINIMAX_DEPTH  = 3

pygame.init()
SCREEN = pygame.display.set_mode(SCREEN_SIZE)
pygame.display.set_caption("Othello / Reversi ~ AI Project")

BIG_FONT   = pygame.font.SysFont("Arial", 48, bold=True)
MED_FONT   = pygame.font.SysFont("Arial", 32)
SMALL_FONT = pygame.font.SysFont("Arial", 24)

############################################
# Othello Tahta Fonksiyonları
############################################
def create_board():
    """
    Başlangıç dizilimi (standart Othello):
       - Orta 4 hücrede 2 siyah, 2 beyaz
    """
    board = np.zeros((ROWS, COLUMNS), dtype=int)
    board[3][3] = AI
    board[3][4] = HUMAN
    board[4][3] = HUMAN
    board[4][4] = AI
    return board

def on_board(r, c):
    """r,c koordinatının tahtada geçerli olup olmadığını döndürür."""
    return 0 <= r < ROWS and 0 <= c < COLUMNS

def get_opponent(player):
    return HUMAN if player == AI else AI

def flips_any(board, row, col, player, opponent):
    """
    Bu pozisyona player taşı koyduğumuzda, en az bir yönde
    rakip taş(lar)ını çeviriyor mu?
    """
    directions = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1)]
    for (dr,dc) in directions:
        if would_flip(board, row, col, player, opponent, dr, dc):
            return True
    return False

def would_flip(board, row, col, player, opponent, dr, dc):
    """
    Belirli (dr,dc) yönünde rakip taşları çevirme durumu var mı?
    """
    r = row + dr
    c = col + dc
    if not on_board(r,c) or board[r][c] != opponent:
        return False

    r += dr
    c += dc
    while on_board(r,c):
        if board[r][c] == EMPTY:
            return False
        if board[r][c] == player:
            return True
        r += dr
        c += dc
    return False

def valid_moves(board, player):
    """
    O anki board'da player'ın yapabileceği
    tüm geçerli (row,col) hamlelerini döndürür.
    """
    opp = get_opponent(player)
    moves = []
    for r in range(ROWS):
        for c in range(COLUMNS):
            if board[r][c] == EMPTY:
                # Boş hücreye koyunca rakip taş çeviriyor mu?
                if flips_any(board, r, c, player, opp):
                    moves.append((r,c))
    return moves

def apply_move(board, row, col, player):
    """
    (row,col)'e player taşı koy,
    çevirme gereken taşları da çevir.
    """
    opponent = get_opponent(player)
    board[row][col] = player
    directions = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1)]
    for (dr,dc) in directions:
        if would_flip(board, row, col, player, opponent, dr, dc):
            flip_direction(board, row, col, player, opponent, dr, dc)

def flip_direction(board, row, col, player, opponent, dr, dc):
    """
    (dr,dc) yönünde rakip taşları çevir.
    """
    r = row+dr
    c = col+dc
    while on_board(r,c) and board[r][c] == opponent:
        r += dr
        c += dc
    # Şimdi board[r][c]==player ise geri dönerek çevir
    if on_board(r,c) and board[r][c] == player:
        r -= dr
        c -= dc
        while (r!=row) or (c!=col):
            board[r][c] = player
            r -= dr
            c -= dc

def board_full(board):
    """Board tamamen dolu mu?"""
    return not np.any(board == EMPTY)

def no_moves_left(board, player):
    """Player'ın hiç hareketi kalmadı mı?"""
    return len(valid_moves(board, player)) == 0

def is_terminal_board(board):
    """
    Oyun bitişi:
      - Tahta dolduysa
      - Ya da her iki taraf da hareket edemiyorsa
    """
    if board_full(board):
        return True
    if no_moves_left(board, HUMAN) and no_moves_left(board, AI):
        return True
    return False

def get_winner(board):
    """
    Oyun bittiğinde, siyah taş sayısı > beyaz ise "Human",
    beyaz taş sayısı > siyah ise "AI",
    eşitse None (berabere).
    """
    human_count = np.count_nonzero(board == HUMAN)
    ai_count    = np.count_nonzero(board == AI)
    if human_count > ai_count:
        return "Human"
    elif ai_count > human_count:
        return "AI"
    return None

############################################
# Board Çizme
############################################
def draw_whole_board(board, turn, game_over, ai_last_move_time, avg_time, total_moves,
                     ai_total_time=0.0, username="", algorithm="", difficulty="Easy"):
    """
    Ekrandaki tahtayı ve sağ paneli çizer.
    ai_last_move_time -> son AI hamlesinin süresi (sıfırlanmasın).
    """
    SCREEN.fill(BLACK)

    # Tahta kareleri (koyu yeşil + beyaz çerçeve)
    for r in range(ROWS):
        for c in range(COLUMNS):
            x_pos = c*SQUARESIZE
            y_pos = r*SQUARESIZE
            pygame.draw.rect(SCREEN, DARK_GREEN, (x_pos, y_pos, SQUARESIZE, SQUARESIZE))
            pygame.draw.rect(SCREEN, WHITE, (x_pos, y_pos, SQUARESIZE, SQUARESIZE),1)

    # Tahtadaki taşlar
    for r in range(ROWS):
        for c in range(COLUMNS):
            piece = board[r][c]
            if piece != EMPTY:
                color = BLACK if piece == HUMAN else WHITE
                cx = c * SQUARESIZE + SQUARESIZE//2
                cy = r * SQUARESIZE + SQUARESIZE//2
                pygame.draw.circle(SCREEN, color, (cx, cy), RADIUS)

    # İnsan sırası ise geçerli hamleleri küçük daireyle göster
    if not game_over and turn == HUMAN:
        moves = valid_moves(board, HUMAN)
        for (rr, cc) in moves:
            cx = cc*SQUARESIZE + SQUARESIZE//2
            cy = rr*SQUARESIZE + SQUARESIZE//2
            pygame.draw.circle(SCREEN, (0,200,0), (cx, cy), 8)

    # Sağ panel
    pygame.draw.rect(SCREEN, GREY, (WIDTH, 0, INFO_WIDTH, HEIGHT))

    # Panel yazıları
    user_text = SMALL_FONT.render(f"Player: {username}", True, BLACK)
    SCREEN.blit(user_text, (WIDTH+20, 10))

    algo_text = SMALL_FONT.render(f"Algorithm: {algorithm}", True, BLACK)
    SCREEN.blit(algo_text, (WIDTH+20, 45))

    diff_text = SMALL_FONT.render(f"Difficulty: {difficulty}", True, BLACK)
    SCREEN.blit(diff_text, (WIDTH+20, 80))

    if not game_over:
        if turn==HUMAN:
            turn_txt = MED_FONT.render("Your Turn (Black)", True, BLACK)
        else:
            turn_txt = MED_FONT.render("AI's Turn (White)", True, BLACK)
    else:
        turn_txt = MED_FONT.render("Game Over", True, BLACK)
    SCREEN.blit(turn_txt, (WIDTH+20, 130))

    # AI Last Move Time
    time_txt = SMALL_FONT.render(f"AI Last Move Time: {ai_last_move_time:.4f}s", True, BLACK)
    SCREEN.blit(time_txt, (WIDTH+20, 170))

    # Avg time
    if avg_time is not None:
        avg_txt = SMALL_FONT.render(f"Avg Time: {avg_time:.4f}s", True, BLACK)
        SCREEN.blit(avg_txt, (WIDTH+20, 195))

    moves_txt = SMALL_FONT.render(f"Total Moves: {total_moves}", True, BLACK)
    SCREEN.blit(moves_txt, (WIDTH+20, 220))

    ai_total_txt = SMALL_FONT.render(f"AI Total Time: {ai_total_time:.4f}s", True, BLACK)
    SCREEN.blit(ai_total_txt, (WIDTH+20, 245))

    # Siyah-beyaz taş sayısı
    human_count = np.count_nonzero(board==HUMAN)
    ai_count    = np.count_nonzero(board==AI)
    cnt_txt = SMALL_FONT.render(f"Black: {human_count} | White: {ai_count}", True, BLACK)
    SCREEN.blit(cnt_txt, (WIDTH+20, 290))

    # Quit butonu
    quit_button = pygame.Rect(WIDTH+20, 340, 120, 40)
    pygame.draw.rect(SCREEN, QUIT_RED, quit_button)
    quit_surf = SMALL_FONT.render("QUIT (Q)", True, WHITE)
    quit_rect = quit_surf.get_rect(center= quit_button.center)
    SCREEN.blit(quit_surf, quit_rect)

    pygame.display.update()

############################################
# Taş Koyma Animasyonu
############################################
def animate_piece_place(board, row, col, player,
                        turn, game_over, ai_last_move_time, avg_time, total_moves,
                        ai_total_time=0.0, username="", algorithm="", difficulty="Easy"):
    """
    Basit bir animasyon: Taş yavaşça büyüyor.
    """
    clock = pygame.time.Clock()
    color = BLACK if player == HUMAN else WHITE
    steps = 12

    for step in range(1, steps+1):
        SCREEN.fill(BLACK)

        # Tahta kareleri
        for rr in range(ROWS):
            for cc in range(COLUMNS):
                x_pos = cc*SQUARESIZE
                y_pos = rr*SQUARESIZE
                pygame.draw.rect(SCREEN, DARK_GREEN, (x_pos, y_pos, SQUARESIZE, SQUARESIZE))
                pygame.draw.rect(SCREEN, WHITE, (x_pos, y_pos, SQUARESIZE, SQUARESIZE),1)

        # Mevcut taşlar
        for rr in range(ROWS):
            for cc in range(COLUMNS):
                piece = board[rr][cc]
                if piece!= EMPTY:
                    clr = BLACK if piece==HUMAN else WHITE
                    cx = cc*SQUARESIZE + SQUARESIZE//2
                    cy = rr*SQUARESIZE + SQUARESIZE//2
                    pygame.draw.circle(SCREEN, clr, (cx, cy), RADIUS)

        # Yeni taş
        cx_new = col*SQUARESIZE + SQUARESIZE//2
        cy_new = row*SQUARESIZE + SQUARESIZE//2
        current_radius = int(RADIUS * (step/steps))
        pygame.draw.circle(SCREEN, color, (cx_new, cy_new), current_radius)

        # Sağ panel
        pygame.draw.rect(SCREEN, GREY, (WIDTH, 0, INFO_WIDTH, HEIGHT))

        user_text = SMALL_FONT.render(f"Player: {username}", True, BLACK)
        SCREEN.blit(user_text, (WIDTH+20, 10))

        algo_text = SMALL_FONT.render(f"Algorithm: {algorithm}", True, BLACK)
        SCREEN.blit(algo_text, (WIDTH+20, 45))

        diff_text = SMALL_FONT.render(f"Difficulty: {difficulty}", True, BLACK)
        SCREEN.blit(diff_text, (WIDTH+20, 80))

        if not game_over:
            if turn==HUMAN:
                turn_txt = MED_FONT.render("Your Turn (Black)", True, BLACK)
            else:
                turn_txt = MED_FONT.render("AI's Turn (White)", True, BLACK)
        else:
            turn_txt = MED_FONT.render("Game Over", True, BLACK)
        SCREEN.blit(turn_txt, (WIDTH+20, 130))

        time_txt= SMALL_FONT.render(f"AI Last Move Time: {ai_last_move_time:.4f}s", True, BLACK)
        SCREEN.blit(time_txt, (WIDTH+20, 170))

        if avg_time is not None:
            avg_txt = SMALL_FONT.render(f"Avg Time: {avg_time:.4f}s", True, BLACK)
            SCREEN.blit(avg_txt, (WIDTH+20, 195))

        moves_txt = SMALL_FONT.render(f"Total Moves: {total_moves}", True, BLACK)
        SCREEN.blit(moves_txt, (WIDTH+20, 220))

        ai_total_txt = SMALL_FONT.render(f"AI Total Time: {ai_total_time:.4f}s", True, BLACK)
        SCREEN.blit(ai_total_txt, (WIDTH+20, 245))

        hc = np.count_nonzero(board==HUMAN)
        ac = np.count_nonzero(board==AI)
        cnt_txt = SMALL_FONT.render(f"Black: {hc} | White: {ac}", True, BLACK)
        SCREEN.blit(cnt_txt, (WIDTH+20, 290))

        # Quit butonu
        quit_button = pygame.Rect(WIDTH+20, 340, 120, 40)
        pygame.draw.rect(SCREEN, QUIT_RED, quit_button)
        quit_surf = SMALL_FONT.render("QUIT (Q)", True, WHITE)
        quit_rect = quit_surf.get_rect(center= quit_button.center)
        SCREEN.blit(quit_surf, quit_rect)

        pygame.display.update()
        clock.tick(60)

def place_piece_animated(board, player, row, col,
                         turn, game_over, ai_last_move_time, avg_time, total_moves,
                         ai_total_time=0.0, username="", algorithm="", difficulty="Easy"):
    """
    apply_move + animasyon
    """
    apply_move(board, row, col, player)
    animate_piece_place(board, row, col, player,
                        turn, game_over, ai_last_move_time, avg_time, total_moves,
                        ai_total_time, username, algorithm, difficulty)

############################################
# Minimax, A*, MCTS, Q-Learning vb.
############################################
def evaluate_board_simple(board, player):
    opp = get_opponent(player)
    player_count = np.count_nonzero(board==player)
    opp_count    = np.count_nonzero(board==opp)
    return player_count - opp_count

def minimax(board, depth, alpha, beta, maximizingPlayer, player_max=AI):
    # Klasik minimax (Othello versiyonu)
    if depth==0 or is_terminal_board(board):
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
            if new_score>value:
                value = new_score
                best_move = (r,c)
            alpha = max(alpha, value)
            if alpha>=beta:
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
            if new_score<value:
                value= new_score
                best_move= (r,c)
            beta= min(beta, value)
            if alpha>=beta:
                break
        return best_move, value

def a_star_move(board, player=AI):
    """
    'A*', en iyi local hamleyi
    (evaluate_board_simple) skoru en büyük yapan seçiyor.
    """
    vm = valid_moves(board, player)
    if not vm:
        return None
    best_move = random.choice(vm)
    best_score= -999999
    for (r,c) in vm:
        temp= board.copy()
        apply_move(temp, r,c, player)
        sc= evaluate_board_simple(temp, player)
        if sc> best_score:
            best_score= sc
            best_move= (r,c)
    return best_move

class MCTSNode:
    """
    MCTS düğümü. Othello rollout mantığı.
    """
    def __init__(self, board, parent=None, current_player=AI, last_move=None):
        self.board= board
        self.parent= parent
        self.current_player= current_player
        self.last_move= last_move

        self.children= []
        self.wins=0
        self.visits=0
        self.untried_moves= valid_moves(board, current_player)

    def is_fully_expanded(self):
        return len(self.untried_moves)==0

    def ucb_score(self, c=1.4142):
        if self.visits==0:
            return float('inf')
        return (self.wins/self.visits) + c* math.sqrt(math.log(self.parent.visits)/ self.visits)

    def select_child(self):
        # En yüksek UCB değerine sahip çocuğu seç
        return max(self.children, key=lambda c: c.ucb_score())

    def expand(self):
        # Eğer untried_moves yoksa pass durumu
        if not self.untried_moves:
            temp= self.board.copy()
            np= get_opponent(self.current_player)
            child= MCTSNode(temp, parent=self, current_player=np, last_move=None)
            self.children.append(child)
            return child

        # Normal
        move= random.choice(self.untried_moves)
        temp= self.board.copy()
        apply_move(temp, move[0], move[1], self.current_player)
        np= get_opponent(self.current_player)
        child= MCTSNode(temp, parent=self, current_player=np, last_move=move)
        self.children.append(child)
        self.untried_moves.remove(move)
        return child

    def rollout(self):
        # Rollout -> rastgele oynayarak bitişe kadar
        temp= self.board.copy()
        cp= self.current_player
        while not is_terminal_board(temp):
            mv= valid_moves(temp, cp)
            if not mv:
                cp= get_opponent(cp)
                if not valid_moves(temp, cp):
                    break
                continue
            chosen= random.choice(mv)
            apply_move(temp, chosen[0], chosen[1], cp)
            cp= get_opponent(cp)

        w= get_winner(temp)
        if w=="AI":
            return 1
        elif w=="Human":
            return -1
        return 0

    def backpropagate(self, result):
        self.visits+=1
        self.wins+= result
        if self.parent:
            self.parent.backpropagate(result)

def mcts_move(board, player=AI, simulations=50):
    vm= valid_moves(board, player)
    if not vm:
        return None
    root= MCTSNode(board.copy(), None, player)
    for _ in range(simulations):
        node= root
        # Selection
        while node.is_fully_expanded() and node.children:
            node= node.select_child()
        # Expansion
        if not is_terminal_board(node.board):
            node= node.expand()
        # Rollout
        result= node.rollout()
        # Backprop
        node.backpropagate(result)

    if not root.children:
        return None
    best_child= max(root.children, key=lambda c: c.visits)
    return best_child.last_move

# Q-Learning basit
def get_state_key(board, player):
    return (tuple(board.flatten()), player)

def get_best_q_action(board, player):
    vm= valid_moves(board, player)
    if not vm:
        return None
    st= get_state_key(board, player)
    if st not in q_table:
        return random.choice(vm)
    best_mv= None
    best_val= -999999
    for mv in vm:
        val= q_table[st].get(mv, 0.0)
        if val> best_val:
            best_val= val
            best_mv= mv
    if best_mv is None:
        return random.choice(vm)
    return best_mv

def q_learning_move(board, player=AI):
    return get_best_q_action(board, player)

############################################
# Negamax Algorithm
############################################

def negamax_algorithm(board, depth, color, player, alpha, beta):
    """
    Negamax Algorithm:
    - board: Oyun tahtası
    - depth: Arama derinliği
    - color: 1 (maximizing perspektifi) veya -1 (minimizing perspektifi)
    - player: Şu an hamle yapan oyuncu
    - alpha, beta: Alfa-beta budaması için sınırlar
    
    Terminal durum veya derinlik 0'da, mevcut board'un heuristic değerini 
    (evaluate_board_simple) 'color' ile çarparak döner.
    """
    if depth == 0 or is_terminal_board(board):
        return None, color * evaluate_board_simple(board, player)
    
    best_move = None
    value = -math.inf
    moves = valid_moves(board, player)
    
    if not moves:
        # Hamle yoksa, pas durumunu simüle etmek için oyuncu değiştirip değeri ters çeviriyoruz.
        return None, -negamax_algorithm(board, depth-1, -color, get_opponent(player), -beta, -alpha)[1]
    
    for move in moves:
        temp = board.copy()
        apply_move(temp, move[0], move[1], player)
        # Negamax: rakip perspektifini elde etmek için değeri ters çevirerek çağırıyoruz.
        _, score = negamax_algorithm(temp, depth-1, -color, get_opponent(player), -beta, -alpha)
        score = -score
        
        if score > value:
            value = score
            best_move = move
        
        alpha = max(alpha, score)
        if alpha >= beta:
            break  # Beta budaması
    
    return best_move, value

############################################
# Advanced Heuristic Based Search
############################################

def coin_parity(board, player):
    opp = get_opponent(player)
    player_count = np.count_nonzero(board == player)
    opp_count = np.count_nonzero(board == opp)
    if player_count + opp_count == 0:
        return 0
    return 100 * (player_count - opp_count) / (player_count + opp_count)

def mobility(board, player):
    opp = get_opponent(player)
    player_moves = len(valid_moves(board, player))
    opp_moves = len(valid_moves(board, opp))
    if player_moves + opp_moves == 0:
        return 0
    return 100 * (player_moves - opp_moves) / (player_moves + opp_moves)

def corners_captured(board, player):
    opp = get_opponent(player)
    corners = [(0,0), (0,COLUMNS-1), (ROWS-1,0), (ROWS-1,COLUMNS-1)]
    player_corners = sum(1 for (r, c) in corners if board[r][c] == player)
    opp_corners = sum(1 for (r, c) in corners if board[r][c] == opp)
    if player_corners + opp_corners == 0:
        return 0
    return 100 * (player_corners - opp_corners) / (player_corners + opp_corners)

def stability(board, player):
    opp = get_opponent(player)
    stable_weight = 1.0
    semi_stable_weight = 0.5
    unstable_weight = 0.1

    def player_stability(p):
        stable = 0
        semi_stable = 0
        unstable = 0
        for r in range(ROWS):
            for c in range(COLUMNS):
                if board[r][c] == p:
                    if (r, c) in [(0,0), (0,COLUMNS-1), (ROWS-1,0), (ROWS-1,COLUMNS-1)]:
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
    # Ağırlıklar: coin parity %25, mobility %25, corners %30, stability %20
    return 0.25 * cp + 0.25 * mob + 0.3 * cor + 0.2 * stab

# Gelişmiş Heuristic Fonksiyonu kullanan Minimax
def minimax_advanced(board, depth, alpha, beta, maximizingPlayer, player_max=AI):
    if depth == 0 or is_terminal_board(board):
        return None, evaluate_board_advanced(board, player_max)
    
    # Hamleleri belirlerken, hangi oyuncunun hamle yapacağına dikkat ediyoruz.
    current_player = player_max if maximizingPlayer else get_opponent(player_max)
    moves = valid_moves(board, current_player)
    if not moves:
        # Pas durumu: Hamle yoksa oyuncu pas geçer.
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
# Iterative Deepening with Time Constraint
############################################

def iterative_deepening_time_move(board, player=AI, time_limit=1.0):
    """
    Iterative Deepening with Time Constraint:
    
    Belirlenen time_limit (saniye cinsinden) dolana kadar,
    arama derinliği kademeli olarak artırılır. Her derinlikte,
    minimax (alfa-beta budamalı) kullanılarak en iyi hamle bulunur.
    
    Eğer zaman sınırı dolarsa, o ana kadar elde edilen en iyi hamle
    geri döndürülür.
    """
    start_time = time.time()
    best_move = None
    depth = 1
    # Zaman dolduğunda mevcut en iyi hamleyi döndürmek için iterative deepening
    while True:
        # Zaman kontrolü
        current_time = time.time()
        if current_time - start_time > time_limit:
            break
        # Şu anki derinlikte minimax araması yapılıyor
        move, score = minimax(board, depth, -math.inf, math.inf, True, player)
        # Aramadan geçerli hamle bulunduysa güncelleniyor
        if move is not None:
            best_move = move
        # Terminal durum veya tahtada hamle kalmamışsa döngüden çıkılır
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
        # Hamle yoksa, pas durumunu simüle ediyoruz.
        return None, minimax_move_ordering(board, depth - 1, alpha, beta, not maximizingPlayer, player_max)[1]
    
    # Transposition table kontrolü
    board_key = (tuple(board.flatten()), current_player)
    if 'transposition_table' not in globals():
        global transposition_table
        transposition_table = {}
    trans_move = transposition_table.get(board_key)
    if trans_move is not None and trans_move in moves:
        moves.remove(trans_move)
        moves = [trans_move] + moves
    else:
        # Hamleleri statik heuristic'e göre sıralıyoruz
        def move_score(move):
            temp = board.copy()
            apply_move(temp, move[0], move[1], current_player)
            return evaluate_board_simple(temp, player_max)
        # Maximizing durumunda yüksek, minimizing'de düşük skorlar öncelikli olsun
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
                break  # Beta budaması
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

############################################
# Zorluk (rastgele hamle)
############################################
def difficulty_to_random_prob(difficulty):
    if difficulty=="Easy":
        return 0.75
    elif difficulty=="Medium":
        return 0.50
    elif difficulty=="Hard":
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
############################################
# CSV Kaydı
############################################
def write_csv_stats(username, algorithm, difficulty, winner,
                    total_moves, ai_total_time, ai_moves, game_duration):
    """
    Oyun bitişinde CSV'ye ekle.
    """
    filename= f"{username}_stats_othello.csv"
    file_exists= os.path.isfile(filename)

    with open(filename, "a", newline="", encoding="utf-8") as csvfile:
        fieldnames= [
            "Username","Algorithm","Difficulty","Winner",
            "TotalMoves","AiMoves","AiThinkingTime","GameDuration"
        ]
        writer= csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        if not file_exists:
            writer.writeheader()

        row_data= {
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

############################################
# Oyun Döngüsü (run_game)
############################################
def run_game(username, algorithm, let_me_win, difficulty):
    """
    Oyunun ana döngüsü.
     - let_me_win => AI pas geçsin.
     - Q tuşu / QUIT butonu => return reason='quit'
     - AI Last Move Time => ai_last_move_time
    """
    board= create_board()
    game_over=False
    turn= random.choice([HUMAN, AI])
    minimax_times=[]
    ai_total_time=0.0
    total_moves=0
    ai_moves=0
    winner=None
    reason=None

    # AI son hamle süresini burada tutalım:
    ai_last_move_time= 0.0

    start_time= time.time()

    while not game_over:
        # Her turda board çiz
        draw_whole_board(
            board= board,
            turn= turn,
            game_over= game_over,
            ai_last_move_time= ai_last_move_time,  # <-- Ekranda gösterilecek
            avg_time= (sum(minimax_times)/len(minimax_times) if minimax_times else None),
            total_moves= total_moves,
            ai_total_time= ai_total_time,
            username= username,
            algorithm= algorithm,
            difficulty= difficulty
        )

        # Event'ler
        for event in pygame.event.get():
            if event.type== pygame.QUIT:
                pygame.quit()
                sys.exit()

            # Q tuşu
            if event.type== pygame.KEYDOWN and event.key== pygame.K_q:
                reason='quit'
                return None, minimax_times, total_moves, ai_total_time, board, reason

            # Mouse
            if event.type== pygame.MOUSEBUTTONDOWN:
                mx,my= event.pos
                # QUIT butonu
                quit_rect= pygame.Rect(WIDTH+20, 340, 120,40)
                if quit_rect.collidepoint(mx,my):
                    reason='quit'
                    return None, minimax_times, total_moves, ai_total_time, board, reason

                # Human hamlesi
                if turn==HUMAN and not game_over:
                    if mx<WIDTH:  # Tahta içine tıklama
                        col= mx//SQUARESIZE
                        row= my//SQUARESIZE
                        vm= valid_moves(board, HUMAN)
                        if (row,col) in vm:
                            # Taş koy
                            place_piece_animated(
                                board, HUMAN, row,col,
                                turn, game_over,
                                ai_last_move_time,  # AI süresi korunur
                                (sum(minimax_times)/len(minimax_times) if minimax_times else None),
                                total_moves,
                                ai_total_time,
                                username,
                                algorithm,
                                difficulty
                            )
                            total_moves+=1
                            # Oyun bitti mi?
                            if is_terminal_board(board):
                                game_over= True
                                winner= get_winner(board)
                            else:
                                # Sıra AI'ya
                                if valid_moves(board, AI):
                                    turn= AI
                                else:
                                    # AI pas
                                    pass

        # AI Hamlesi
        if not game_over and turn==AI:
            vm= valid_moves(board, AI)
            if not vm:
                # Pas
                turn= HUMAN
            else:
                pygame.time.wait(300)  # ufak bir bekleme
                start_t= time.time()
                best_mv= ai_move(board, let_me_win=let_me_win, algorithm=algorithm, difficulty=difficulty)
                run_t= time.time()- start_t

                if best_mv is None:
                    # Yani let_me_win => pas
                    turn= HUMAN
                else:
                    # Normal hamle
                    minimax_times.append(run_t)
                    ai_total_time += run_t
                    ai_moves+=1
                    total_moves+=1

                    # Son AI hamlesinin süresi
                    ai_last_move_time= run_t

                    (rr,cc)= best_mv
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
                        game_over= True
                        winner= get_winner(board)
                    else:
                        # İnsan pas?
                        if not valid_moves(board, HUMAN):
                            pass
                        else:
                            turn= HUMAN

        if is_terminal_board(board):
            game_over= True
            winner= get_winner(board)

    end_time= time.time()
    game_duration= end_time- start_time

    # CSV
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
# Board Çizme (Eğitimde kullanılan sade sürüm)
############################################
def draw_board_training_visual(board, turn, game_over):
    """
    Q-Learning eğitimi sırasında her hamlede tahtayı kısaca göstereceğiz.
    Bu, animasyonsuz, basit bir görselleştirme.
    """
    SCREEN.fill(BLACK)

    # Tahta kareleri
    for r in range(ROWS):
        for c in range(COLUMNS):
            x_pos = c*SQUARESIZE
            y_pos = r*SQUARESIZE
            pygame.draw.rect(SCREEN, DARK_GREEN, (x_pos,y_pos,SQUARESIZE,SQUARESIZE))
            pygame.draw.rect(SCREEN, WHITE, (x_pos,y_pos,SQUARESIZE,SQUARESIZE),1)

    # Taşlar
    for r in range(ROWS):
        for c in range(COLUMNS):
            piece = board[r][c]
            if piece!=EMPTY:
                color= BLACK if piece==HUMAN else WHITE
                cx= c*SQUARESIZE+ SQUARESIZE//2
                cy= r*SQUARESIZE+ SQUARESIZE//2
                pygame.draw.circle(SCREEN, color, (cx,cy), RADIUS)

    # Sağ tarafa basit metin
    pygame.draw.rect(SCREEN, GREY, (WIDTH,0,INFO_WIDTH,HEIGHT))
    if not game_over:
        status_txt= MED_FONT.render("Training...", True, BLACK)
    else:
        status_txt= MED_FONT.render("Episode Over", True, BLACK)
    SCREEN.blit(status_txt, (WIDTH+ 20, 10))

    pygame.display.update()

############################################
# Q-Learning Parametreli Minimax
############################################
def minimax_training_opponent(board):
    """
    Eğitimde, AI'nin rakibi Minimax (TRAIN_MINIMAX_DEPTH) olsun.
    """
    from math import inf
    vm= valid_moves(board, HUMAN)
    if not vm:
        return None
    # Basit minimax (depth=TRAIN_MINIMAX_DEPTH)
    # Minimax kodunu kısaca ekliyorum:
    best_mv, _= minimax(board, TRAIN_MINIMAX_DEPTH, -inf, inf, False, AI) 
    return best_mv

############################################
# Q-Learning Fonksiyonları
############################################
def get_state_key(board, player):
    return (tuple(board.flatten()), player)

def get_best_q_action(board, player):
    moves= valid_moves(board, player)
    if not moves:
        return None
    st= get_state_key(board, player)
    if st not in q_table:
        return random.choice(moves)
    # En yüksek Q'yu veren hamleyi bul
    best_val= -999999
    best_mv= None
    for mv in moves:
        val= q_table[st].get(mv, 0.0)
        if val> best_val:
            best_val= val
            best_mv= mv
    if best_mv is None:
        return random.choice(moves)
    return best_mv

def train_q_learning_visual_custom(episodes=10, alpha=0.1, gamma=0.95,
                                   epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.99):
    """
    Othello Q-Learning eğitimi (visual):
      - AI (Q-Learning) vs. Minimax (depth=TRAIN_MINIMAX_DEPTH)
      - Ekranda her hamleyi göster
      - Q tablosu saklanır.

    Parametreler:
      episodes: Kaç oyun
      alpha:   Öğrenme hızı
      gamma:   Gelecek ödül indirgeme
      epsilon: Epsilon-greedy başlangıç
      epsilon_min
      epsilon_decay
    """
    global q_table
    ep= 0
    clock= pygame.time.Clock()

    print("[TRAIN VISUAL] Starting Q-Learning... Press Q to stop early.")

    while ep< episodes:
        board= create_board()
        game_over= False
        turn= random.choice([HUMAN, AI])  # Rastgele kim başlasın
        # Epsilon-greedy
        while not game_over:
            # Her turda ekrana tahta çiz
            draw_board_training_visual(board, turn, game_over)
            pygame.time.wait(30)
            clock.tick(30)

            for event in pygame.event.get():
                if event.type== pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type== pygame.KEYDOWN:
                    if event.key== pygame.K_q:
                        # Erken çık, tabloyu kaydet
                        with open("q_table_othello.pkl","wb") as f:
                            pickle.dump(q_table, f)
                        print(f"[TRAIN VISUAL] Early exit. {ep} episodes done.")
                        return

            if turn== AI:
                # Q-learning AI
                st= get_state_key(board, AI)
                if st not in q_table:
                    q_table[st]= {}

                moves= valid_moves(board, AI)
                if not moves:
                    # Pas
                    turn= HUMAN
                    # Eğer human da yoksa game over
                    if not valid_moves(board, HUMAN):
                        game_over= True
                    continue

                # Epsilon-greedy seçim
                if random.random()< epsilon:
                    action= random.choice(moves)
                else:
                    # En iyi Q
                    best_val= -999999
                    action= None
                    for mv in moves:
                        val= q_table[st].get(mv, 0.0)
                        if val>best_val:
                            best_val= val
                            action= mv

                # Uygula
                row,col= action
                board[row][col]= AI
                apply_move(board, row,col, AI)

                # Ödül hesapla
                # - Othello'da anlık ödül=0, eğer kazandıysak=1, kaybettiysek=-1, vs.
                #   Terminal durum mu diye bak
                reward= 0.0
                if is_terminal_board(board):
                    w= get_winner(board)
                    if w=="AI":
                        reward= 1.0
                    elif w=="Human":
                        reward= -1.0
                    game_over= True

                # Q güncelleme
                new_st= get_state_key(board, HUMAN)
                if new_st not in q_table:
                    q_table[new_st]= {}

                old_q= q_table[st].get(action, 0.0)
                if not game_over:
                    # max future
                    fut_moves= valid_moves(board, HUMAN)
                    if fut_moves:
                        max_future= max(q_table[new_st].get(mv,0.0) for mv in fut_moves)
                    else:
                        max_future= 0
                else:
                    max_future= 0

                new_q= old_q + alpha*(reward + gamma*max_future - old_q)
                q_table[st][action]= new_q

                if not game_over:
                    turn= HUMAN

            else:
                # Rakip = HUMAN (aslında Minimax)
                # minimax ile hamle yapsın
                mv= valid_moves(board, HUMAN)
                if not mv:
                    turn= AI
                    if not valid_moves(board, AI):
                        game_over= True
                    continue
                best_mv= minimax_training_opponent(board)  # Depth=TRAIN_MINIMAX_DEPTH
                if best_mv is None:
                    # bir hata yoksa random hamle
                    best_mv= random.choice(mv)
                # hamleyi uygula
                r,c= best_mv
                board[r][c]= HUMAN
                apply_move(board, r,c, HUMAN)
                if is_terminal_board(board):
                    # Kazandı mı?
                    w= get_winner(board)
                    if w=="Human":
                        # AI için reward -1 vs. 
                        pass
                    game_over= True
                else:
                    turn= AI

        # episode bitti
        ep+= 1
        if epsilon> epsilon_min:
            epsilon*= epsilon_decay
        print(f"[TRAIN VISUAL] Episode {ep}/{episodes} done. Epsilon={epsilon:.3f}")

    # Tüm eğitim bitti
    with open("q_table_othello.pkl","wb") as f:
        pickle.dump(q_table, f)
    print("[TRAIN VISUAL] All training done. q_table_othello.pkl saved.")

############################################
# Game Over & End Screen
############################################
def show_game_over_screen():
    """
    Oyun biter bitmez ekrana yarı saydam bir "Game Over" yazısı
    """
    overlay = pygame.Surface((WIDTH+INFO_WIDTH, HEIGHT))
    overlay.fill((0,0,0))
    overlay.set_alpha(180)
    txt= BIG_FONT.render("Game Over", True, WHITE)
    txt_rect= txt.get_rect(center=((WIDTH+INFO_WIDTH)//2, HEIGHT//2))
    SCREEN.blit(overlay, (0,0))
    SCREEN.blit(txt, txt_rect)
    pygame.display.update()
    pygame.time.wait(1500)

def show_end_screen(winner, total_moves, ai_total_time, board):
    """
    Winner'a göre ekrana yazı. "Play again?" -> YES/NO
    """
    SCREEN.fill((50,50,50))
    lines=[]
    hc= np.count_nonzero(board==HUMAN)
    ac= np.count_nonzero(board==AI)

    if winner=="AI":
        lines= [
            "AI (White) wins!",
            f"Score: (Black) {hc} - (White) {ac}",
            f"AI Thinking Time: {ai_total_time:.4f}s",
            f"Total Moves: {total_moves}",
            "Play again?"
        ]
        color= WHITE
    elif winner=="Human":
        lines= [
            "You (Black) win!",
            f"Score: (Black) {hc} - (White) {ac}",
            f"Total Moves: {total_moves}",
            "Play again?"
        ]
        color= BLACK
    else:
        lines= [
            "It's a tie!",
            f"Score: (Black) {hc} - (White) {ac}",
            f"Total Moves: {total_moves}",
            "Play again?"
        ]
        color= GREY

    line_height= 50
    total_text_height= len(lines)* line_height
    start_y= (HEIGHT- total_text_height)//2
    center_x= (WIDTH+INFO_WIDTH)//2

    for i,line in enumerate(lines):
        surf= BIG_FONT.render(line, True, color)
        rect= surf.get_rect(center=(center_x, start_y + i*line_height))
        SCREEN.blit(surf, rect)

    yes_button= pygame.Rect(center_x-120, start_y + len(lines)* line_height +30, 80,40)
    no_button = pygame.Rect(center_x+40,  start_y + len(lines)* line_height +30, 80,40)

    pygame.draw.rect(SCREEN, GREY, yes_button)
    pygame.draw.rect(SCREEN, GREY, no_button)

    yes_text= SMALL_FONT.render("YES", True, BLACK)
    no_text = SMALL_FONT.render("NO", True, BLACK)

    yes_rect= yes_text.get_rect(center=yes_button.center)
    no_rect = no_text.get_rect(center=no_button.center)

    SCREEN.blit(yes_text, yes_rect)
    SCREEN.blit(no_text, no_rect)

    pygame.display.update()
    clock= pygame.time.Clock()
    while True:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type== pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type== pygame.MOUSEBUTTONDOWN:
                if yes_button.collidepoint(event.pos):
                    return True
                elif no_button.collidepoint(event.pos):
                    return False

############################################
# Menü Fonksiyonları
############################################
def show_username_screen():
    """
    En ilk menü. Burada 'Exit' butonu koyduk (Back yerine).
    """
    clock = pygame.time.Clock()
    input_box = pygame.Rect((WIDTH+INFO_WIDTH)//2 -150, HEIGHT//2, 300,40)
    username= ""
    active_text= False
    play_button= pygame.Rect((WIDTH+INFO_WIDTH)//2 -75, input_box.y+60, 150,40)

    exit_button= pygame.Rect(20, HEIGHT-60, 100,40)  # 'Exit'

    while True:
        clock.tick(30)
        SCREEN.fill((50,50,50))

        title_surf= BIG_FONT.render("Othello / Reversi AI", True, WHITE)
        title_rect= title_surf.get_rect(center=((WIDTH+INFO_WIDTH)//2, HEIGHT//4))
        SCREEN.blit(title_surf, title_rect)

        prompt_surf= SMALL_FONT.render("Enter your username:", True, WHITE)
        SCREEN.blit(prompt_surf, (input_box.x, input_box.y-30))

        pygame.draw.rect(SCREEN, WHITE, input_box,2)
        name_surf= SMALL_FONT.render(username, True, WHITE)
        SCREEN.blit(name_surf, (input_box.x+5, input_box.y+5))

        pygame.draw.rect(SCREEN, GREY, play_button)
        p_surf= SMALL_FONT.render("Play", True, BLACK)
        p_rect= p_surf.get_rect(center= play_button.center)
        SCREEN.blit(p_surf, p_rect)

        # Exit butonu
        pygame.draw.rect(SCREEN, ORANGE, exit_button)
        e_surf= SMALL_FONT.render("Exit", True, BLACK)
        e_rect= e_surf.get_rect(center= exit_button.center)
        SCREEN.blit(e_surf, e_rect)

        pygame.display.update()

        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                pygame.quit()
                sys.exit()

            elif event.type==pygame.MOUSEBUTTONDOWN:
                # Exit
                if exit_button.collidepoint(event.pos):
                    pygame.quit()
                    sys.exit()

                # Input box
                if input_box.collidepoint(event.pos):
                    active_text= True
                else:
                    active_text= False

                # Play
                if play_button.collidepoint(event.pos):
                    if username.strip()=="":
                        username="player"
                    return username

            elif event.type==pygame.KEYDOWN:
                if active_text:
                    if event.key==pygame.K_RETURN:
                        if username.strip()=="":
                            username="player"
                        return username
                    elif event.key==pygame.K_BACKSPACE:
                        username= username[:-1]
                    else:
                        username+= event.unicode

def show_algorithm_screen():
    """
    Algoritma seçimi menüsü.
    "Choose algorithm" başlığı ekranın üstünde sabitlenir,
    algoritma butonları başlık altından başlayıp sabit bir alanda sıralanır,
    "Let me win" ve "Next" butonları ise ekranın alt kısmında yer alır.
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
    start_y = 100  # Algoritma butonları bu y koordinatından başlayacak
    algo_buttons = []
    chosen_algo = "Min-Max with Alpha-Beta Pruning"
    let_me_win = False

    center_x = (WIDTH + INFO_WIDTH) // 2
    button_width = 400  # Buton genişliğini 400 olarak ayarlıyoruz

    # Back butonu sol alt köşede
    back_button = pygame.Rect(20, HEIGHT - 60, 100, 40)

    # Algoritma butonlarını oluşturuyoruz (ortalanmış)
    for i, txt in enumerate(button_texts):
        rect = pygame.Rect(center_x - button_width // 2, start_y + i * (button_height + gap), button_width, button_height)
        algo_buttons.append((rect, txt))

    # "Let me win" butonunu alt kısma ortalayalım
    let_me_win_button = pygame.Rect(center_x - button_width // 2, HEIGHT - 120, button_width, 40)
    # "Next" butonunu alt kısımda sabit tutuyoruz (genişliği 150)
    next_button = pygame.Rect(center_x - 75, HEIGHT - 60, 150, 40)

    title_y = 50  # Üstteki başlık konumu

    while True:
        clock.tick(30)
        SCREEN.fill((50, 50, 50))

        # Üstte sabit başlık
        prompt_surf = MED_FONT.render("Choose algorithm:", True, WHITE)
        prompt_rect = prompt_surf.get_rect(center=(center_x, title_y))
        SCREEN.blit(prompt_surf, prompt_rect)

        # Algoritma butonları
        for (rct, text) in algo_buttons:
            col = ORANGE if text == chosen_algo else GREY
            pygame.draw.rect(SCREEN, col, rct)
            t_surf = SMALL_FONT.render(text, True, BLACK)
            t_rect = t_surf.get_rect(center=rct.center)
            SCREEN.blit(t_surf, t_rect)

        # "Let me win" butonu alt kısımda sabit
        lw_col = ORANGE if let_me_win else GREY
        pygame.draw.rect(SCREEN, lw_col, let_me_win_button)
        lw_txt = SMALL_FONT.render("Let me win", True, BLACK)
        lw_rect = lw_txt.get_rect(center=let_me_win_button.center)
        SCREEN.blit(lw_txt, lw_rect)

        # "Next" butonu alt kısımda sabit
        pygame.draw.rect(SCREEN, GREY, next_button)
        n_surf = SMALL_FONT.render("Next", True, BLACK)
        n_rect = n_surf.get_rect(center=next_button.center)
        SCREEN.blit(n_surf, n_rect)

        # Back butonu sol alt köşede
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
    """
    Q-Learning train -> episode sayısı gir. 
    'Back' butonu ile geri dön.
    """
    clock= pygame.time.Clock()
    input_box= pygame.Rect((WIDTH+INFO_WIDTH)//2 -150, HEIGHT//2, 300,40)
    user_input=""
    active_text=False
    train_button= pygame.Rect((WIDTH+INFO_WIDTH)//2 -75, input_box.y+60, 150,40)
    back_button= pygame.Rect(20, HEIGHT-60, 100,40)

    while True:
        clock.tick(30)
        SCREEN.fill((50,50,50))

        title_surf= MED_FONT.render("Enter number of episodes:", True, WHITE)
        title_rect= title_surf.get_rect(center=((WIDTH+INFO_WIDTH)//2, HEIGHT//4))
        SCREEN.blit(title_surf, title_rect)

        pygame.draw.rect(SCREEN, WHITE, input_box,2)
        ep_surf= SMALL_FONT.render(user_input, True, WHITE)
        SCREEN.blit(ep_surf, (input_box.x+5, input_box.y+5))

        pygame.draw.rect(SCREEN, GREY, train_button)
        t_surf= SMALL_FONT.render("Train", True, BLACK)
        t_rect= t_surf.get_rect(center=train_button.center)
        SCREEN.blit(t_surf, t_rect)

        # Back
        pygame.draw.rect(SCREEN, ORANGE, back_button)
        b_surf= SMALL_FONT.render("Back", True, BLACK)
        b_rect= b_surf.get_rect(center= back_button.center)
        SCREEN.blit(b_surf, b_rect)

        pygame.display.update()

        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type==pygame.MOUSEBUTTONDOWN:
                if input_box.collidepoint(event.pos):
                    active_text=True
                else:
                    active_text=False

                if train_button.collidepoint(event.pos):
                    if user_input.strip()=="":
                        return 10, None
                    else:
                        return int(user_input), None

                if back_button.collidepoint(event.pos):
                    return None, 'back'

            elif event.type==pygame.KEYDOWN:
                if active_text:
                    if event.key==pygame.K_RETURN:
                        if user_input.strip()=="":
                            return 10, None
                        else:
                            return int(user_input), None
                    elif event.key==pygame.K_BACKSPACE:
                        user_input= user_input[:-1]
                    elif event.unicode.isdigit():
                        user_input+= event.unicode

def show_difficulty_screen():
    """
    Zorluk seçimi. Back butonu ile algoritma menüsüne dönüyor.
    """
    clock= pygame.time.Clock()
    difficulties= ["Easy","Medium","Hard","Extreme"]
    button_height=50
    gap=20
    start_y= HEIGHT//2 -120
    diff_buttons=[]
    chosen_diff="Easy"

    back_button= pygame.Rect(20, HEIGHT-60, 100,40)

    for i, text in enumerate(difficulties):
        rect= pygame.Rect((WIDTH+INFO_WIDTH)//2 -150, start_y + i*(button_height+gap), 300, button_height)
        diff_buttons.append((rect, text))

    start_button= pygame.Rect((WIDTH+INFO_WIDTH)//2 -75, start_y +4*(button_height+gap) +50, 150,40)

    while True:
        clock.tick(30)
        SCREEN.fill((50,50,50))

        prompt_surf= MED_FONT.render("Choose difficulty:", True, WHITE)
        prompt_rect= prompt_surf.get_rect(center=((WIDTH+INFO_WIDTH)//2, HEIGHT//4))
        SCREEN.blit(prompt_surf, prompt_rect)

        for (rct, dt) in diff_buttons:
            col= ORANGE if dt==chosen_diff else GREY
            pygame.draw.rect(SCREEN, col, rct)
            dt_surf= SMALL_FONT.render(dt, True, BLACK)
            dt_rect= dt_surf.get_rect(center= rct.center)
            SCREEN.blit(dt_surf, dt_rect)

        pygame.draw.rect(SCREEN, GREY, start_button)
        st_surf= SMALL_FONT.render("Start Game", True, BLACK)
        st_rect= st_surf.get_rect(center=start_button.center)
        SCREEN.blit(st_surf, st_rect)

        # Back
        pygame.draw.rect(SCREEN, ORANGE, back_button)
        b_surf= SMALL_FONT.render("Back", True, BLACK)
        b_rect= b_surf.get_rect(center= back_button.center)
        SCREEN.blit(b_surf, b_rect)

        pygame.display.update()

        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type==pygame.MOUSEBUTTONDOWN:
                for (rct, dt_) in diff_buttons:
                    if rct.collidepoint(event.pos):
                        chosen_diff= dt_
                if start_button.collidepoint(event.pos):
                    return chosen_diff, None
                if back_button.collidepoint(event.pos):
                    return None, 'back'

############################################
# main
############################################
def main():
    """
    Tüm menü akışı, Q-Learning train (opsiyon), difficulty, vs.
    Oyun bitince "Play again?" -> Evet/Hayır.
    'Let me win' -> AI pas, 
    AI Last Move Time -> saklıyoruz.
    """
    global q_table

    # Q-table yükle
    if os.path.exists("q_table_othello.pkl"):
        with open("q_table_othello.pkl","rb") as f:
            q_table = pickle.load(f)
        print("[INFO] q_table_othello.pkl loaded.")
    else:
        print("[INFO] No q_table_othello.pkl found. Starting fresh...")
        q_table = {}

    # 1) Username screen
    username= show_username_screen()

    while True:
        # 2) Algoritma ekranı
        while True:
            algorithm, let_me_win, reason_algo= show_algorithm_screen()
            if reason_algo=='back':
                # Geri -> username ekranı
                username= show_username_screen()
            else:
                break

        # 2b) Train Q-Learning (Visual) mi?
        if algorithm=="Train Q-Learning (Visual)":
            while True:
                ep_count, reason_ep = show_episodes_screen_for_visual()
                if reason_ep=='back':
                    # geri algoya
                    break
                else:
                    # train
                    print(f"[INFO] Visual Q-Learning train for {ep_count} episodes.")
                    train_q_learning_visual_custom(episodes= ep_count)
                    break
            continue

        # 3) Difficulty ekranı
        while True:
            difficulty, reason_diff= show_difficulty_screen()
            if reason_diff=='back':
                # geri -> algorithm
                break
            if difficulty is not None:
                # 4) Oyunu başlat
                winner, minimax_times, total_moves, ai_total_time, final_board, reason_game= run_game(
                    username, algorithm, let_me_win, difficulty
                )
                if reason_game=='quit':
                    print("[INFO] QUIT in game -> returning to algo screen.")
                    break
                # Normal bitiş
                show_game_over_screen()
                play_again= show_end_screen(winner, total_moves, ai_total_time, final_board)
                if not play_again:
                    pygame.quit()
                    sys.exit()
                # Evet -> tekrarla (burada yine diff menüsüne, ister en başa)
                break
        # Bura difficulty while
        # Tekrar algorithm'e dönebilir

    pygame.quit()
    sys.exit()

if __name__=="__main__":
    main()

import random, math
from board import valid_moves, apply_move, is_terminal_board, get_winner, get_opponent

class MCTSNode:
    def __init__(self, board, parent=None, current_player=2, last_move=None):   #root node
        self.board = board
        self.parent = parent
        self.current_player = current_player
        self.last_move = last_move
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = valid_moves(board, current_player)
    
    def is_fully_expanded(self):    # denenmeyen hamle kaldı mı? 
        return len(self.untried_moves) == 0

    def ucb_score(self, c=1.4142):  # childin selection score unu hesaplar
        if self.visits == 0:
            return float('inf')
        return (self.wins / self.visits) + c * math.sqrt(math.log(self.parent.visits) / self.visits)

    def select_child(self): # ucb skorlara göre child i seçer
        return max(self.children, key=lambda c: c.ucb_score())

    def expand(self):   # denenmemiş hamleleri dener ve yeni child node oluşturur
        if not self.untried_moves:  # eğer hamle kalmadıysa boş hamle yapıp ağacı ilerletir ve yeni child oluşturur
            temp = self.board.copy()
            next_player = get_opponent(self.current_player)
            child = MCTSNode(temp, parent=self, current_player=next_player, last_move=None)
            self.children.append(child)
            return child
        move = random.choice(self.untried_moves)
        temp = self.board.copy()
        apply_move(temp, move[0], move[1], self.current_player)
        next_player = get_opponent(self.current_player)
        child = MCTSNode(temp, parent=self, current_player=next_player, last_move=move)
        self.children.append(child)
        self.untried_moves.remove(move)
        return child

    def rollout(self):  # current node dan rastgele bir hamle yapar ve oyunu oynatır
        temp = self.board.copy()
        cp = self.current_player
        while not is_terminal_board(temp):
            moves = valid_moves(temp, cp)
            if not moves:
                cp = get_opponent(cp)
                if not valid_moves(temp, cp):
                    break
                continue
            chosen = random.choice(moves)
            apply_move(temp, chosen[0], chosen[1], cp)
            cp = get_opponent(cp)
        w = get_winner(temp)
        if w == "AI":
            return 1
        elif w == "Human":
            return -1
        return 0

    def backpropagate(self, result):    # simulation sonucunu geri yayar 
        self.visits += 1
        self.wins += result
        if self.parent:
            self.parent.backpropagate(result)

def mcts_move(board, player, simulations=50):   # main fonk
    moves = valid_moves(board, player)
    if not moves:
        return None
    root = MCTSNode(board.copy(), None, player)
    for _ in range(simulations):
        node = root
        while node.is_fully_expanded() and node.children:
            node = node.select_child()
        if not is_terminal_board(node.board):
            node = node.expand()
        result = node.rollout()
        node.backpropagate(result)
    if not root.children:
        return None
    best_child = max(root.children, key=lambda c: c.visits)
    return best_child.last_move

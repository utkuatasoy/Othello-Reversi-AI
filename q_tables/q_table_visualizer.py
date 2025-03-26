import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pickle
import time

def plot_board_with_q(board, moves_dict, player):
    """
    board: 8x8 numpy dizisi, 0: boş, 1: Siyah, 2: Beyaz
    moves_dict: o state için q_table'daki hamle -> Q değeri sözlüğü
    player: Bu state için geçerli oyuncu
    """
    # Board durumuna göre renkler: 0 => koyu yeşil, 1 => siyah, 2 => beyaz
    cmap = mcolors.ListedColormap(["#007800", "black", "white"])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(6, 6))
    plt.imshow(board, cmap=cmap, norm=norm, interpolation='none')
    cbar = plt.colorbar(ticks=[0, 1, 2])
    cbar.set_label('Tahta Durumu (0: boş, 1: Siyah, 2: Beyaz)')
    plt.title(f"Board State (Player: {player})")
    plt.xlabel("Kolon")
    plt.ylabel("Satır")
    plt.xticks(np.arange(8))
    plt.yticks(np.arange(8))
    plt.grid(which='major', color='gray', linestyle='-', linewidth=1)
    
    # Q değeri bilgisi varsa ilgili hücreye overlay yapalım
    for (r, c), q_value in moves_dict.items():
        plt.text(c, r, f"{q_value:.2f}", ha="center", va="center", color="red",
                 fontsize=12, fontweight="bold")
    
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(1)
    plt.close()

def show_all_state_heatmaps(q_table):
    """
    q_table içindeki her state için, tahtanın durumunu ve varsa Q değeri bilgisini
    1 saniye arayla gösterir.
    """
    for key, moves_dict in q_table.items():
        board_flat, player = key
        board = np.array(board_flat).reshape(8, 8)
        
        # Konsola state ve Q değerlerini yazdır
        print("=" * 50)
        print(f"State (Player: {player}):")
        print(board)
        print("Q Değerleri:")
        if moves_dict:
            for move, q in moves_dict.items():
                print(f"  Hamle {move}: {q}")
        else:
            print("  (Q değeri bulunamadı)")
        print("=" * 50)
        
        # Board durumunu ve Q değerlerini görselleştir
        plot_board_with_q(board, moves_dict, player)

if __name__ == "__main__":
    # q_table_othello.pkl dosyasını yükle
    try:
        with open("q_table_othello.pkl", "rb") as f:
            q_table = pickle.load(f)
        print("[INFO] q_table_othello.pkl başarıyla yüklendi.\n")
    except FileNotFoundError:
        print("[INFO] q_table_othello.pkl bulunamadı. Lütfen eğitim tamamlandıktan sonra bu dosyayı oluşturduğunuzdan emin olun.")
        q_table = {}
    
    # Tüm state'ler için görselleştirmeyi sırayla göster
    show_all_state_heatmaps(q_table)
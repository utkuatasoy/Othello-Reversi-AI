# Othello / Reversi: AI Algorithms Project

Bu repo, Othello/Reversi oyununun Yapay Zeka tabanlı bir versiyonunu içerir. Proje kapsamında Minimax (Alpha-Beta Pruning dâhil), A*, Monte Carlo Tree Search (MCTS), Q-Learning, Negamax, gelişmiş sezgisel arama, Iterative Deepening ve Move Ordering gibi çeşitli algoritmalar kullanılmıştır. Ayrıca, kullanıcıya zorluk (Easy, Medium, Hard, Extreme) ve algoritma seçme imkânı sunulmuştur. Bu sayede, oyunun stratejik derinliği ve oynanabilirliği artırılmıştır.

Proje raporunda (aşağıdaki özet bilgilerde) bu algoritmaların matematiksel temelleri, uygulama detayları, Q-Learning eğitim süreci ile farklı stratejilerin avantaj ve dezavantajları ele alınmıştır.

## İçindekiler
- [Genel Bakış](#genel-bakış)
- [Algoritmalar](#algoritmalar)
  - [Minimax (Alpha-Beta Pruning)](#minimax-alpha-beta-pruning)
  - [A* Algoritması](#a-algoritması)
  - [Monte Carlo Tree Search (MCTS)](#monte-carlo-tree-search-mcts)
  - [Q-Learning](#q-learning)
  - [Negamax](#negamax)
  - [Gelişmiş Sezgisel Tabanlı Arama](#gelişmiş-sezgisel-tabanlı-arama)
  - [Iterative Deepening with Time Constraint](#iterative-deepening-with-time-constraint)
  - [Move Ordering](#move-ordering)
- [Zorluk Seviyeleri](#zorluk-seviyeleri)
- [Kurulum ve Kullanım](#kurulum-ve-kullanım)
- [Kod Yapısı](#kod-yapısı)
- [Sonuçlar ve Değerlendirme](#sonuçlar-ve-değerlendirme)
- [Kaynaklar](#kaynaklar)

---

## Genel Bakış

Othello (ya da Reversi), 8x8’lik bir tahtada iki oyuncunun (Siyah ve Beyaz) stratejik hamle yaparak rakibin taşlarını çevirmeye çalıştığı klasik bir oyundur. Bu projede temel hedef, çeşitli yapay zeka algoritmaları kullanarak oyunu oynamak ve bu algoritmaların performanslarını (kullanım kolaylığı, hesaplama süresi, stratejik derinlik) karşılaştırmaktır.

Projede öne çıkan özellikler:
- **Birden fazla AI algoritması** seçeneği (Minimax, A*, MCTS, vb.).
- **Q-Learning eğitimi** (Minimax rakibe karşı görsel ya da başlat/bitir kontrollü).
- **Zorluk seviyesi** seçimi (Easy, Medium, Hard, Extreme) ve buna bağlı **rastgele hamle oranı**.
- **Oyun istatistikleri**nin (toplam hamle, AI süreleri vb.) CSV dosyasına kaydedilmesi.
- **Oyun sonrası "Play again?"** seçeneği ile tekrar oynama veya çıkış.
- **Taş koyma animasyonları** ve **geçerli hamle vurgulama** (kullanıcı dostu arayüz).

---

## Algoritmalar

### Minimax (Alpha-Beta Pruning)
- İki oyunculu sıfır toplamlı oyunlarda klasik strateji belirleme yöntemi.
- Alpha-Beta budama (\(\alpha\) ve \(\beta\) değerleri) gereksiz dalları keserek performansı artırır.
- Proje kodunda `minimax` fonksiyonu ve derinlik parametresi ile uygulanmıştır.

### A* Algoritması
- Başlangıçtan hedefe en kısa/optimal yol arayan bir yöntem.
- \(g(n) + h(n)\) formülü ile sıradaki durumu seçer; bu projede basit bir sezgisel (\(h\)) ile uygulanmıştır.
- Kapsamlı durum uzayında, uygun (veya uygun olmayan) heuristik fonksiyonlara bağlı olarak avantaj/dezavantajı vardır.

### Monte Carlo Tree Search (MCTS)
- Rastgele simülasyonlar (rollout) ile en umut vaat eden dalı seçen, dört aşamalı (selection, expansion, simulation, backpropagation) bir algoritma.
- UCB (\( \frac{w_i}{n_i} + c \sqrt{\frac{\ln N}{n_i}} \)) formülüyle keşif-sömürü dengesini kurar.
- Geniş arama alanlarında etkili ancak simülasyon maliyeti artabilir.

### Q-Learning
- Takviyeli öğrenme yöntemi: Durum-hamle çiftlerine ait Q-değerleri iteratif olarak güncellenir.
- \[
Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \right]
\]
- Projede, Q tablosu (\(q\_table\)) saklanarak her hamle güncellenir, Minimax rakibiyle eğitim yapılır ve istenirse görsel eğitim modu mevcuttur.

### Negamax
- Minimax’in simetrik bir varyantı: \(\max -V(m)\) ile tek formül üzerinden çözüm.
- Kod sadeliğini artırır ancak karmaşık oyun durumu varyasyonları için bazen esnekliği azalabilir.

### Gelişmiş Sezgisel Tabanlı Arama
- **Coin Parity**, **Mobility**, **Corners Captured**, **Stability** gibi çok boyutlu değerlendirme ölçütleri.
- Değerlendirme fonksiyonu:
  \[
  0.25 \times \text{Coin Parity} + 0.25 \times \text{Mobility} + 0.30 \times \text{Corners} + 0.20 \times \text{Stability}
  \]
- Daha stratejik hamleler getirebilir ama hesaplama maliyeti yüksektir.

### Iterative Deepening with Time Constraint
- Belirlenen zaman sınırı (örneğin 1.0s) içinde derinliği kademeli artırır.
- “Anytime” özelliği ile sürekli kullanılabilecek en iyi hamleyi elde tutar; ancak her katmanda tam arama yapmak ek maliyet yaratır.

### Move Ordering
- Alpha-Beta budamada hamleleri en çok değerli görülenlerden en az değerlilere göre sıralayarak budamanın etkinliğini artırır.
- Statik değerlendirmenin hatalı olması durumunda yanlış sıralama olabilir.

---

## Zorluk Seviyeleri
Oyunda zorluk seviyesi seçildiğinde, AI’nın belli oranda rastgele hamle yapması sağlanarak oyuncunun deneyimi çeşitlendirilir:
- **Easy**: \%75 rastgele
- **Medium**: \%50 rastgele
- **Hard**: \%25 rastgele
- **Extreme**: Hiç rastgele hamle yok, tamamen deterministik seçim

Bu oranlar `difficulty_to_random_prob` fonksiyonunda tanımlanmıştır.

---

## Kurulum ve Kullanım

1. **Projeyi klonlayın** (veya zip olarak indirin):
   ```bash
   git clone https://github.com/kullaniciadi/othello-reversi-ai.git
   cd othello-reversi-ai
   ```

2. **Gerekli kütüphaneleri yükleyin** (öneri: sanal ortam içinde):
   ```bash
   pip install pygame numpy
   ```
   (Ek olarak `pickle`, `time`, `csv`, `os` vb. Python standart kütüphaneler kullanılmaktadır.)

3. **Projeyi çalıştırın**:
   ```bash
   python main.py
   ```
   - Kullanıcı adı girildikten sonra algoritma seçimi menüsü gelecektir.
   - “Train Q-Learning (Visual)” seçerek görsel olarak Q-learning eğitimi yapılabilir.
   - Zorluk (Easy/Medium/Hard/Extreme) seçimi sonrası oyun başlar.

4. **Oyun süresince**:
   - Siyah (Human) taşlar için geçerli hamleler yeşil bir noktayla belirtilir.
   - Q tuşu ile oyunu erken kapatabilirsiniz (csv kaydı vs. verileri tutar).
   - Oyun bittiğinde skorları görebilir, “Play again?” üzerinden yeni oyuna geçebilir veya çıkabilirsiniz.

5. **CSV Kaydı**:
   - Oyun sonlandığında, `kullaniciadi_stats_othello.csv` benzeri bir dosyaya istatistikler (toplam hamle, AI süreleri, kazanan vb.) eklenir.
   - Q-Learning tablosu `q_table_othello.pkl` dosyasında saklanır; sonraki çalıştırmalarda bu tablo yüklenerek aynı öğrenme devam ettirilir.

---


---

## Sonuçlar ve Değerlendirme

### Genel Bulgular
- **Minimax (Alpha-Beta)** derinlik parametresiyle en iyi hamleleri bulma konusunda etkili, ancak çok geniş arama uzayında süre uzayabiliyor.
- **A*** daha kısa vadeli ve heuristik odaklı, uygun olmayan durumda optimalden sapabilir.
- **MCTS** özellikle belirsizlik ve geniş durum uzayı olan senaryolarda iyi sonuç verir, ancak simülasyon sayısı arttıkça zaman artar.
- **Q-Learning** ek bir eğitim aşaması gerektirir, uzun vadede benzer durumlarda daha iyi hamle yapma avantajı sunar.
- **Negamax** daha sade bir Minimax varyantı. Aynı performansa yakın sonuçlar verir.
- **Gelişmiş Sezgisel** (coin parity, mobility, corners, stability) ile durum analizi daha derin ancak işlem yükü artar.
- **Iterative Deepening** zaman sınırlı ortamlarda faydalı (“anytime search”), tekrar arama nedeniyle maliyetli olabilir.
- **Move Ordering** alpha-beta budamasının etkinliğini arttırır, ancak statik değerlendirme hataları olması dezavantajdır.
- **Zorluk Seviyeli Rastgelelik** kullanıcı deneyimini çeşitlendirmede başarılıdır.

Bu algoritmalar arasından seçim yaparak, oyunun stratejik derinliğini ve hesaplama hızını kullanıcı ihtiyacına göre dengelemek mümkündür.

---

## Kaynaklar

1. C. J. C. H. Watkins and P. Dayan, “Q-learning,” *Machine Learning*, vol. 8, no. 3, pp. 279–292, 1992.  
2. S. Russell and P. Norvig, *Artificial Intelligence: A Modern Approach*, 3rd ed. Prentice Hall, 2010.  
3. M. Buro, “Improving heuristic mini-max search by supervised learning,” *Proc. of the 2002 Conference on Games in AI Research*, pp. 85–99, 2002.  
4. M. van der Ree and M. Wiering, “Reinforcement learning in the game of Othello: Learning against a fixed opponent and learning from self-play,” in *Proc. of ADPRL*, 2013.  
5. A. Norelli and A. Panconesi, “OLIVAW: Mastering Othello without Human Knowledge, nor a Fortune,” *arXiv preprint arXiv:2103.17228*, 2021.  
6. C. Browne et al., “A survey of Monte Carlo tree search methods,” *IEEE Transactions on Computational Intelligence and AI in Games*, vol. 4, no. 1, pp. 1–43, 2012.  
7. R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed. MIT Press, 2018.  
8. M. L. Littman, “Value-function reinforcement learning in Markov games,” *Journal of Cognitive Systems Research*, vol. 2, no. 1, pp. 55–66, 2001.  
9. J. Scheiermann and W. Konen, “AlphaZero-Inspired Game Learning: Faster Training by Using MCTS Only at Test Time,” *arXiv preprint arXiv:2204.13307*, 2022.  

---


Herhangi bir geri bildiriminiz veya sorunuz varsa, lütfen issue açın veya pull request gönderin.

Teşekkürler!
```

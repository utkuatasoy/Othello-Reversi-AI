# Othello / Reversi: AI Algorithms Project

This repository contains an AI-powered version of the classic Othello/Reversi game. Various algorithms such as Minimax (with Alpha-Beta Pruning), A*, Monte Carlo Tree Search (MCTS), Q-Learning, Negamax, Advanced Heuristic Search, Iterative Deepening, and Move Ordering have been implemented. Users can choose both algorithm and difficulty level (Easy, Medium, Hard, Extreme), allowing for enhanced strategic depth and dynamic gameplay.

The detailed report (see summary below) covers the mathematical foundations, implementation strategies, Q-Learning training procedure, and pros/cons of each algorithm, supported by visuals.

---

## Table of Contents

- [Overview](#overview)
- [Algorithms](#algorithms)
- [Difficulty Levels](#difficulty-levels)
- [Screenshots](#screenshots)
- [Installation & Usage](#installation--usage)
- [Code Structure](#code-structure)
- [Results & Evaluation](#results--evaluation)
- [References](#references)

---

## Overview

Othello is a strategic 8x8 board game where two players (Black and White) try to outmaneuver each other by flipping the opponent’s pieces. The primary goal of this project is to simulate intelligent gameplay using different AI algorithms and analyze their performance in terms of execution time, strategic quality, and learning ability.

Key Features:
- Multiple AI algorithms (Minimax, A*, MCTS, etc.)
- Visual and interactive Q-Learning training
- Adjustable difficulty settings with controlled randomness
- Logging game stats to CSV
- "Play Again?" option at game end
- Move animations and legal move highlights

---

## Algorithms

(Algorithms section remains unchanged — see your previous content)

---

## Difficulty Levels

(Randomness explained — unchanged from previous content)

---

## 📷 Screenshots

Below are screenshots from various parts of the project interface:

### 🎮 Game Screens

- **In-Game Board View**  
  ![Game Board](figure/oyun_ici.png)  
  Displays the game board with black/white pieces, current player information, legal move highlights, and status panel.

- **Alternate In-Game View**  
  ![Game Board 2](figure/oyun_ici1.png)  
  A second perspective on the game board emphasizing visual elements like highlighted moves and player stats.

### ⚙️ Interface Elements

- **Algorithm Selection Screen**  
  ![Algorithm Selection](figure/algoritma_secim.png)  
  User selects the desired AI algorithm (Minimax, A*, MCTS, etc.).

- **Difficulty Selection Screen**  
  ![Difficulty Selection](figure/zorluk_secim.png)  
  User chooses between Easy, Medium, Hard, and Extreme difficulty levels.

### 🧠 Q-Learning Training

- **Episode Count Input Screen**  
  ![Q-Learning Episode](figure/qlearning_episode.png)  
  Interface to input how many episodes the agent should train against the Minimax opponent.

- **Q-Learning Training Interface**  
  ![Q-Learning Training](figure/qlearning_training.png)  
  Shows Q-Learning agent in training mode, playing against a Minimax opponent.

- **Q-Table Sample**  
  ![Q-Table](figure/qtable.png)  
  Displays a snapshot of state-action Q-values learned by the agent.

- **Q-Table Plot**  
  ![Q-Table Plot](figure/qtable1.png)  
  Visual representation of the learned Q-values over training.

### 🏁 Game Over Screens

- **AI Victory Screen**  
  ![Game Over - AI](figure/oyun_sonu.png)  
  Shows endgame statistics when AI wins.

- **Player Victory / Tie Screen**  
  ![Game Over - Player](figure/oyun_sonu1.png)  
  Shows final result when the player wins or a tie occurs.

### 📊 Performance Evaluation

- **AI Win Rate by Algorithm**  
  ![Win Rate Chart](figure/kazanim_oranlari_grafik.png)  
  Displays win percentages for each algorithm.

- **Thinking Time vs. Win Rate**  
  ![Thinking Time vs Win Rate](figure/thinking_time.png)  
  Comparison between average AI thinking time and win rate.

- **Average Game Duration**  
  ![Average Duration](figure/rank.png)  
  Illustrates average duration per game per algorithm.

- **Games Played and AI Wins**  
  ![Game Count vs Wins](figure/download.png)  
  Total number of games played and AI win counts by algorithm.

---

## Installation & Usage

(Unchanged from your original, but you can optionally adjust "Private" under `git clone` if this is public.)

---

## Results & Evaluation

(Unchanged from your original — contains insightful breakdown of each method’s performance.)

---

## References

(Unchanged)

---

Feel free to open an issue or pull request for questions, bugs, or contributions.

Thanks for visiting!

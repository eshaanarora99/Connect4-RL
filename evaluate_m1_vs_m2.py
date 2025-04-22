import numpy as np
import random
from tensorflow.keras.models import load_model
from gameplay import update_board, check_for_win, find_legal
from tqdm import trange

def choose_action(model, board):
    """Sample from the model’s softmax outputs over legal moves."""
    board_input = np.expand_dims(board, axis=0)
    probs = model.predict(board_input, verbose=0)[0]
    legal = find_legal(board)
    masked = np.zeros_like(probs)
    masked[legal] = probs[legal]
    if masked.sum() == 0:
        masked[legal] = 1.0
    masked /= masked.sum()
    return np.random.choice(len(masked), p=masked)

def evaluate(model_A, model_B, n_games=200):
    wins_A = wins_B = ties = 0
    total_moves = []
    
    for _ in trange(n_games, desc="Evaluating games"):
        board = np.zeros((6, 7, 2), dtype=int)
        current = random.choice(["plus", "minus"])
        moves = 0
        
        while True:
            legal = find_legal(board)
            if not legal:
                ties += 1
                break
            
            actor = model_A if current == "plus" else model_B
            move = choose_action(actor, board)
            board = update_board(board, current, move)
            moves += 1
            
            result = check_for_win(board, move)
            if result != "nobody":
                if result == "plus":
                    wins_A += 1
                else:
                    wins_B += 1
                break
            
            current = "minus" if current == "plus" else "plus"
        
        total_moves.append(moves)
    
    print("--- Evaluation Summary ---")
    print(f"Games: {n_games}")
    print(f"M1 (PG‑trained) wins: {wins_A}")
    print(f"M2 (Nico) wins:      {wins_B}")
    print(f"Ties:                {ties}")
    print(f"Win rate:   {wins_A/n_games:.2%}")
    print(f"Loss rate:  {wins_B/n_games:.2%}")
    print(f"Tie rate:   {ties/n_games:.2%}")
    print(f"Avg moves/game: {np.mean(total_moves):.2f} ± {np.std(total_moves):.2f}")

if __name__ == "__main__":
    model_A = load_model(r"C:\Users\eshaa\Downloads\Opti_Project_3\M1_PG_trained.h5")
#    model_B = load_model(r"C:\Users\eshaa\Downloads\Opti_Project_3\Models\M2.h5") # Original M2 without training
    model_B = load_model(r"C:\Users\eshaa\Downloads\Opti_Project_3\Models\PG and DQN Models\M2_pg.h5") # M2 trained with PG
    evaluate(model_A, model_B, n_games=200)

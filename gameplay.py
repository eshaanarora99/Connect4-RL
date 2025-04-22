import numpy as np
import random
from tensorflow.keras.models import load_model

# ─── Environment ───────────────────────────────────────────────────────────────

def update_board(board, color, column):
    """Drop a piece of 'color' into 'column'."""
    b = board.copy()
    nrow = b.shape[0]
    filled = int(b[:, column, :].sum())
    row = nrow - filled - 1
    if row >= 0:
        if color == "plus":
            b[row, column, 0], b[row, column, 1] = 1, 0
        else:
            b[row, column, 0], b[row, column, 1] = 0, 1
    return b

def find_legal(board):
    """Return list of non‑full columns 0–6."""
    return [c for c in range(board.shape[1]) if board[0, c, :].sum() == 0]

def check_for_win(board, col):
    """
    After a move in column col, check if that created a four‑in‑a‑row.
    Returns "plus", "minus", or "nobody".
    """
    nrow, ncol, _ = board.shape
    filled = int(board[:, col, :].sum())
    row = nrow - filled
    for layer, player in [(0, "plus"), (1, "minus")]:
        # vertical
        if row+3 < nrow and all(board[row+i, col, layer] for i in range(4)):
            return player
        # horizontal
        for start in range(max(0, col-3), min(col+1, ncol-3)):
            if all(board[row, start+i, layer] for i in range(4)):
                return player
        # diag ↗
        for offset in range(-3, 1):
            coords = [(row+offset+i, col+offset+i) for i in range(4)]
            if all(0 <= r < nrow and 0 <= c < ncol and board[r, c, layer] for r, c in coords):
                return player
        # diag ↘
        for offset in range(-3, 1):
            coords = [(row-offset-i, col+offset+i) for i in range(4)]
            if all(0 <= r < nrow and 0 <= c < ncol and board[r, c, layer] for r, c in coords):
                return player
    return "nobody"

# ─── AI Move Selection ────────────────────────────────────────────────────────

def choose_action(model, board, evaluation_mode="sample"):
    """
    If evaluation_mode=="sample", treat model outputs as softmax probs.
    If "greedy", pick highest‑scoring legal move.
    """
    probs = model.predict(np.expand_dims(board, axis=0), verbose=0)[0]
    legal = find_legal(board)
    
    if evaluation_mode == "sample":
        masked = np.zeros_like(probs)
        masked[legal] = probs[legal]
        if masked.sum() == 0:
            masked[legal] = 1.0
        masked /= masked.sum()
        return np.random.choice(len(masked), p=masked)
    
    # greedy
    scores = np.full_like(probs, -np.inf)
    scores[legal] = probs[legal]
    return int(np.argmax(scores))

# ─── Human vs AI Gameplay ─────────────────────────────────────────────────────

def display_board(board):
    """ASCII print the 6×7 board: X=plus, O=minus, .=empty."""
    for row in board:
        print(" ".join("X" if cell[0] else "O" if cell[1] else "." for cell in row))
    print("0 1 2 3 4 5 6\n")

def play_against_model(model):
    board = np.zeros((6,7,2), dtype=int)
    first = input("Play first or second? (1=first, 2=second): ").strip()
    human = "plus" if first=="1" else "minus"
    ai     = "minus" if human=="plus" else "plus"
    turn   = "plus"
    
    while True:
        display_board(board)
        legal = find_legal(board)
        if not legal:
            print("Tie! Board is full.")
            return
        
        if turn == human:
            # human move
            move = None
            while move not in legal:
                try:
                    move = int(input(f"Your move {legal}: "))
                except:
                    pass
        else:
            # AI move
            move = choose_action(model, board, evaluation_mode="sample")
            print(f"AI plays column {move}")
        
        board = update_board(board, turn, move)
        if check_for_win(board, move) != "nobody":
            display_board(board)
            winner = "You" if turn == human else "AI"
            print(f"{winner} win!")
            return
        
        turn = "minus" if turn=="plus" else "plus"

if __name__ == "__main__":
    m = load_model(r"C:\Users\eshaa\Downloads\Opti_Project_3\M1_PG_trained.h5")
    print("Loaded PG‑trained model.\n")
    play_against_model(m)

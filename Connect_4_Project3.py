import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, InputLayer
from tensorflow.keras.optimizers import Adam
import time
import matplotlib.pyplot as plt

# ------------------------------
# Check for CUDA-enabled GPU and configure memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU detected. Using GPU for computations.")
    except RuntimeError as e:
        print("Error setting up GPU memory growth:", e)
else:
    print("No GPU detected. Using CPU for computations.")

############################
# Connect 4 Environment Functions
############################

def update_board(board_temp, color, column):
    board = board_temp.copy()
    nrow, ncol, _ = board.shape
    colsum = board[:, column, 0].sum() + board[:, column, 1].sum()
    row = int(nrow - colsum - 1)
    if row >= 0:
        if color == "plus":
            board[row, column, 0] = 1  # Mark on plus layer
            board[row, column, 1] = 0
        elif color == "minus":
            board[row, column, 1] = 1  # Mark on minus layer
            board[row, column, 0] = 0
    return board

def check_for_win(board, col):
    nrow, ncol, _ = board.shape
    colsum = board[:, col, 0].sum() + board[:, col, 1].sum()
    row = int(nrow - colsum)
    for layer, player in zip([0, 1], ["plus", "minus"]):
        # Vertical win
        if row + 3 < nrow:
            if all(board[row + i, col, layer] == 1 for i in range(4)):
                return f"v-{player}"
        # Horizontal win
        for c_start in range(max(0, col - 3), min(col + 1, ncol - 3)):
            if all(board[row, c_start + i, layer] == 1 for i in range(4)):
                return f"h-{player}"
        # Diagonal (bottom-left to top-right)
        for i in range(-3, 1):
            if all(0 <= row + i + j < nrow and 0 <= col + i + j < ncol and board[row + i + j, col + i + j, layer] == 1 for j in range(4)):
                return f"d-{player}"
        # Diagonal (top-left to bottom-right)
        for i in range(-3, 1):
            if all(0 <= row - i - j < nrow and 0 <= col + i + j < ncol and board[row - i - j, col + i + j, layer] == 1 for j in range(4)):
                return f"d-{player}"
    return "nobody"

def find_legal(board):
    return [col for col in range(board.shape[1])
            if board[0, col, 0] == 0 and board[0, col, 1] == 0]

############################
# Policy Gradient Training Functions (Steps 2–3)
############################

def sample_action(model, board, temperature=1.0):
    board_input = np.expand_dims(board, axis=0)  # shape: (1,6,7,2)
    probs = model.predict(board_input)[0]  # Expected shape: (7,) softmax probabilities
    legal = find_legal(board)
    masked_probs = np.zeros_like(probs)
    masked_probs[legal] = probs[legal]
    # If the sum is zero (e.g. numerical issues), assign uniform probabilities.
    if masked_probs.sum() == 0:
        masked_probs[legal] = 1.0
    masked_probs /= masked_probs.sum()
    action = np.random.choice(len(masked_probs), p=masked_probs)
    log_prob = np.log(masked_probs[action] + 1e-10)
    return action, log_prob

def compute_discounted_rewards(rewards, gamma=0.99):
    discounted = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        discounted.insert(0, R)
    return np.array(discounted)

def play_game_pg(model_M1, model_M2, starting_player="plus"):
    board = np.zeros((6, 7, 2))
    traj = []
    current_color = starting_player
    while True:
        legal = find_legal(board)
        if not legal:
            outcome = 0  # tie
            break
        if current_color == "plus":
            action, log_prob = sample_action(model_M1, board)
            traj.append((board.copy(), action, log_prob))
        else:
            action, _ = sample_action(model_M2, board)
        board = update_board(board, current_color, action)
        result = check_for_win(board, action)
        if result != "nobody":
            outcome = 1 if "plus" in result else -1
            break
        current_color = "minus" if current_color == "plus" else "plus"
    rewards = [outcome] * len(traj)
    discounted_rewards = compute_discounted_rewards(rewards)
    return traj, discounted_rewards

def train_policy_gradient(model_M1, model_M2, episodes=1000, gamma=0.99, learning_rate=1e-3):
    optimizer = Adam(learning_rate)
    all_losses = []
    for ep in range(episodes):
        starting = random.choice(["plus", "minus"])
        traj, discounted_rewards = play_game_pg(model_M1, model_M2, starting_player=starting)
        if len(traj) == 0:
            continue
        states = np.array([s for (s, a, lp) in traj])
        actions = np.array([a for (s, a, lp) in traj])
        returns = discounted_rewards
        with tf.GradientTape() as tape:
            preds = model_M1(states, training=True)  # Shape: (batch, 7)
            indices = tf.stack([tf.range(tf.shape(preds)[0]), actions], axis=1)
            chosen_action_probs = tf.gather_nd(preds, indices)
            log_probs = tf.math.log(chosen_action_probs + 1e-10)
            loss = -tf.reduce_mean(returns * log_probs)
        grads = tape.gradient(loss, model_M1.trainable_variables)
        optimizer.apply_gradients(zip(grads, model_M1.trainable_variables))
        all_losses.append(loss.numpy())
        if (ep + 1) % 100 == 0:
            print(f"PG Episode {ep+1}/{episodes}, Loss: {loss.numpy():.4f}")
    return all_losses

############################
# DQN Training Functions (Step 4)
############################

def build_dqn_model(input_shape=(6,7,2), output_size=7):
    model = Sequential([
        InputLayer(input_shape=input_shape),
        Conv2D(32, kernel_size=3, activation='relu', padding='same'),
        Conv2D(64, kernel_size=3, activation='relu', padding='same'),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(output_size, activation='linear')  # Q-values for each column
    ])
    model.compile(optimizer=Adam(1e-3), loss='mse')
    return model

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
    def add(self, transition):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)
    def sample(self, batch_size):
        return random.sample(self.buffer, min(len(self.buffer), batch_size))

def play_game_dqn(model, opponent_model, epsilon=0.1, gamma=0.99):
    board = np.zeros((6,7,2))
    transitions = []
    current_color = "plus"  # DQN agent as "plus"
    while True:
        legal = find_legal(board)
        if not legal:
            done = True
            reward = 0
            transitions.append((board.copy(), None, reward, board.copy(), done))
            break
        if current_color == "plus":
            if np.random.rand() < epsilon:
                action = random.choice(legal)
            else:
                state_input = np.expand_dims(board, axis=0)
                q_values = model.predict(state_input)[0]
                masked_q = np.full_like(q_values, -np.inf)
                masked_q[legal] = q_values[legal]
                action = np.argmax(masked_q)
            prev_board = board.copy()
            board = update_board(board, "plus", action)
            result = check_for_win(board, action)
            if result != "nobody":
                reward = 1 if "plus" in result else -1
                done = True
            else:
                reward = 0
                done = False
            transitions.append((prev_board, action, reward, board.copy(), done))
        else:
            action, _ = sample_action(opponent_model, board)
            board = update_board(board, "minus", action)
            result = check_for_win(board, action)
            if result != "nobody":
                transitions[-1] = (transitions[-1][0], transitions[-1][1], -1, board.copy(), True)
                return transitions
        current_color = "minus" if current_color == "plus" else "plus"
    return transitions

def train_dqn(model, opponent_model, episodes=1000, batch_size=32, gamma=0.99,
              epsilon_start=1.0, epsilon_min=0.1, epsilon_decay=0.995):
    buffer = ReplayBuffer(10000)
    optimizer = Adam(1e-3)
    losses = []
    epsilon = epsilon_start
    for ep in range(episodes):
        transitions = play_game_dqn(model, opponent_model, epsilon=epsilon, gamma=gamma)
        for t in transitions:
            buffer.add(t)
        batch = buffer.sample(batch_size)
        if len(batch) == 0:
            continue
        states = np.array([s for (s, a, r, s_next, done) in batch])
        actions = np.array([a if a is not None else 0 for (s, a, r, s_next, done) in batch])
        rewards = np.array([r for (s, a, r, s_next, done) in batch])
        next_states = np.array([s_next for (s, a, r, s_next, done) in batch])
        dones = np.array([done for (s, a, r, s_next, done) in batch])
        target_q = model.predict(next_states)
        max_q = np.max(target_q, axis=1)
        targets = rewards + gamma * max_q * (1 - dones.astype(int))
        with tf.GradientTape() as tape:
            q_values = model(states, training=True)
            indices = tf.stack([tf.range(tf.shape(q_values)[0]), actions], axis=1)
            chosen_q = tf.gather_nd(q_values, indices)
            loss = tf.reduce_mean(tf.square(targets - chosen_q))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        losses.append(loss.numpy())
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        if (ep+1) % 100 == 0:
            print(f"DQN Episode {ep+1}/{episodes}, Loss: {loss.numpy():.4f}, Epsilon: {epsilon:.3f}")
    return losses

############################
# New Action Selection Function for Evaluation
############################

def choose_action(model, board, evaluation_mode="sample"):
    """
    Chooses an action based on the model's outputs.
    If evaluation_mode is "sample", uses sampling from softmax outputs (PG models).
    If evaluation_mode is "greedy", uses argmax over model outputs (DQN models).
    """
    if evaluation_mode == "sample":
        return sample_action(model, board)[0]
    elif evaluation_mode == "greedy":
        board_input = np.expand_dims(board, axis=0)
        outputs = model.predict(board_input)[0]
        legal = find_legal(board)
        masked = np.full_like(outputs, -np.inf)
        masked[legal] = outputs[legal]
        return np.argmax(masked)
    else:
        raise ValueError("Invalid evaluation_mode. Choose 'sample' or 'greedy'.")

############################
# Evaluation Functions (Step 5: Comparison)
############################

def evaluate_battle(model_A, model_B, n_games=100, mode_A="sample", mode_B="sample"):
    """
    Evaluate head-to-head battles between two models.
    model_A plays as "plus" and model_B as "minus".
    Returns win counts for A, win counts for B, tie count, average moves, std. dev. of moves, and the list of moves per game.
    """
    wins_A = 0
    wins_B = 0
    ties = 0
    moves_list = []
    for _ in range(n_games):
        board = np.zeros((6,7,2))
        current_color = random.choice(["plus", "minus"])
        moves = 0
        while True:
            legal = find_legal(board)
            if not legal:
                ties += 1
                moves_list.append(moves)
                break
            if current_color == "plus":
                action = choose_action(model_A, board, evaluation_mode=mode_A)
            else:
                action = choose_action(model_B, board, evaluation_mode=mode_B)
            board = update_board(board, current_color, action)
            moves += 1
            result = check_for_win(board, action)
            if result != "nobody":
                if "plus" in result:
                    wins_A += 1
                elif "minus" in result:
                    wins_B += 1
                moves_list.append(moves)
                break
            current_color = "minus" if current_color == "plus" else "plus"
    avg_moves = np.mean(moves_list) if moves_list else 0
    std_moves = np.std(moves_list) if moves_list else 0
    return wins_A, wins_B, ties, avg_moves, std_moves, moves_list

def plot_evaluation_results(results, labels, title):
    n = len(results)
    win_rates_A = [r[0] / sum(r) for r in results]
    win_rates_B = [r[1] / sum(r) for r in results]
    ties_rates = [r[2] / sum(r) for r in results]
    
    x = np.arange(n)
    width = 0.25
    plt.figure(figsize=(8, 5))
    plt.bar(x - width, win_rates_A, width, label='Model A wins')
    plt.bar(x, win_rates_B, width, label='Model B wins')
    plt.bar(x + width, ties_rates, width, label='Ties')
    plt.xticks(x, labels)
    plt.ylabel("Proportion")
    plt.title(title)
    plt.legend()
    plt.show()

############################
# Main Routine: Load Models, Train, Evaluate, and Produce Additional Graphs
############################

if __name__ == "__main__":
    # Load your pre-trained models M1 and M2 (update paths as needed).
    model_M1 = load_model(r"C:\Users\eshaa\Downloads\Opti_Project_3\M1.h5")
    model_M2 = load_model(r"C:\Users\eshaa\Downloads\Opti_Project_3\M2.h5")
    
    # ----- Step 2–3: Policy Gradient Training -----
    print("Starting Policy Gradient Training on M1 (self-play vs M2)...")
    pg_losses = train_policy_gradient(model_M1, model_M2, episodes=500, gamma=0.99, learning_rate=1e-3)
    model_M1.save("M1_PG_trained.h5")
    print("PG training complete. Updated M1 saved as 'M1_PG_trained.h5'.")
    
    # ----- Step 4: DQN Training -----
    print("Starting DQN Training...")
    dqn_model = build_dqn_model(input_shape=(6,7,2), output_size=7)
    dqn_losses = train_dqn(dqn_model, model_M2, episodes=500, batch_size=32, gamma=0.99)
    dqn_model.save("DQN_trained.h5")
    print("DQN training complete. Model saved as 'DQN_trained.h5'.")
    
    # ----- Step 5: Evaluate and Compare Models -----
    # Evaluate PG-trained M1 vs baseline M2 using "sample" mode
    wins_M1, wins_M2, ties, avg_moves_pg, std_moves_pg, moves_list_pg = evaluate_battle(
        model_M1, model_M2, n_games=200, mode_A="sample", mode_B="sample")
    print("\n--- Evaluation: PG Model (M1) vs M2 ---")
    print(f"Games: 200")
    print(f"M1 wins: {wins_M1}, M2 wins: {wins_M2}, Ties: {ties}")
    print(f"Win rate: {wins_M1/200:.2%}, Loss rate: {wins_M2/200:.2%}, Tie rate: {ties/200:.2%}")
    print(f"Average moves per game: {avg_moves_pg:.2f} ± {std_moves_pg:.2f}")
    
    # Evaluate DQN model vs baseline M2 using "greedy" for DQN
    wins_DQN, wins_M2_dqn, ties_dqn, avg_moves_dqn, std_moves_dqn, moves_list_dqn = evaluate_battle(
        dqn_model, model_M2, n_games=200, mode_A="greedy", mode_B="sample")
    print("\n--- Evaluation: DQN vs M2 ---")
    print(f"Games: 200")
    print(f"DQN wins: {wins_DQN}, M2 wins: {wins_M2_dqn}, Ties: {ties_dqn}")
    print(f"Win rate: {wins_DQN/200:.2%}, Loss rate: {wins_M2_dqn/200:.2%}, Tie rate: {ties_dqn/200:.2%}")
    print(f"Average moves per game: {avg_moves_dqn:.2f} ± {std_moves_dqn:.2f}")
    
    # Plot win/tie rates comparison
    results = [(wins_M1, wins_M2, ties), (wins_DQN, wins_M2_dqn, ties_dqn)]
    labels = ["PG vs M2", "DQN vs M2"]
    plot_evaluation_results(results, labels, "Win-Tie Rates for Agents vs. M2")
    
    # ---- Additional Graphs ----
    # Plot training loss curves for PG and DQN
    plt.figure()
    plt.plot(pg_losses, label="PG Losses")
    plt.title("Policy Gradient Training Losses")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.plot(dqn_losses, label="DQN Losses", color="orange")
    plt.title("DQN Training Losses")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    
    # Plot histograms of moves per game for each evaluation
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(moves_list_pg, bins=20, color='blue', alpha=0.7)
    plt.title("Histogram of Moves per Game (PG vs M2)")
    plt.xlabel("Number of Moves")
    plt.ylabel("Frequency")
    
    plt.subplot(1, 2, 2)
    plt.hist(moves_list_dqn, bins=20, color='green', alpha=0.7)
    plt.title("Histogram of Moves per Game (DQN vs M2)")
    plt.xlabel("Number of Moves")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()
    
    print("\nTraining and evaluation complete. Review the printed summary statistics and graphs above for a detailed comparison.")

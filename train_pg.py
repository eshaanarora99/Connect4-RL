# train_pg.py
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from gameplay import update_board, find_legal, check_for_win, choose_action

def sample_action(model, board, temperature=1.0):
    board_input = np.expand_dims(board, axis=0)
    probs = model.predict(board_input, verbose=0)[0]
    legal = find_legal(board)
    masked = np.zeros_like(probs)
    masked[legal] = probs[legal]
    if masked.sum() == 0:
        masked[legal] = 1.0
    masked /= masked.sum()
    action = np.random.choice(len(masked), p=masked)
    log_prob = np.log(masked[action] + 1e-10)
    return action, log_prob

def compute_discounted_rewards(rewards, gamma=0.99):
    discounted, R = [], 0
    for r in reversed(rewards):
        R = r + gamma * R
        discounted.insert(0, R)
    return np.array(discounted)

def play_game_pg(model_M1, model_M2, starting_player="plus"):
    board = np.zeros((6,7,2), dtype=int)
    traj = []
    current = starting_player
    while True:
        legal = find_legal(board)
        if not legal:
            outcome = 0; break
        if current == "plus":
            action, logp = sample_action(model_M1, board)
            traj.append((board.copy(), action, logp))
        else:
            action, _ = sample_action(model_M2, board)
        board = update_board(board, current, action)
        res = check_for_win(board, action)
        if res != "nobody":
            outcome = 1 if res == "plus" else -1
            break
        current = "minus" if current == "plus" else "plus"
    rewards = [outcome]*len(traj)
    return traj, compute_discounted_rewards(rewards)

def train_policy_gradient(model_M1, model_M2, episodes=500, gamma=0.99, lr=1e-3):
    optimizer = Adam(lr)
    losses = []
    for ep in range(episodes):
        start = random.choice(["plus","minus"])
        traj, disc_rewards = play_game_pg(model_M1, model_M2, start)
        if not traj: continue
        states = np.array([s for (s,a,lp) in traj])
        actions = np.array([a for (s,a,lp) in traj])
        returns = disc_rewards
        with tf.GradientTape() as tape:
            preds = model_M1(states, training=True)
            idx = tf.stack([tf.range(tf.shape(preds)[0]), actions], axis=1)
            probs = tf.gather_nd(preds, idx)
            loss = -tf.reduce_mean(returns * tf.math.log(probs + 1e-10))
        grads = tape.gradient(loss, model_M1.trainable_variables)
        optimizer.apply_gradients(zip(grads, model_M1.trainable_variables))
        losses.append(loss.numpy())
        if (ep+1) % 100 == 0:
            print(f"PG Episode {ep+1}/{episodes} Loss: {loss.numpy():.4f}")
    return losses

if __name__ == "__main__":
    m1 = load_model("M1.h5")
    m2 = load_model("M2.h5")
    pg_losses = train_policy_gradient(m1, m2, episodes=500)
    m1.save("M1_PG_trained.h5")
    np.save("pg_losses.npy", pg_losses)
    print("Saved M1_PG_trained.h5 and pg_losses.npy")


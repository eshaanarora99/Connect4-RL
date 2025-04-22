# Optimization II Project III - Connect 4 Reinforcement Learning

## Connect_4_Project3.py

Legacy monolithic script containing all training, evaluation, and plotting logic in one file. Retained for reference but now modularized.

## train_pg.py

Trains the M1 model using policy gradient reinforcement learning in self-play against M2. Saves the improved model as M1_PG_trained.h5.

## train_dqn.py

Trains a separate DQN (Deep Q-Network) agent against M2. Saves the trained agent as DQN_trained.h5.

## evaluate_m1_vs_m2.py

Evaluates head-to-head performance between any two models (e.g., M1 vs M2). Prints win/loss/tie statistics and average move count.

## gameplay.py

Allows a human to play interactively against any trained model in the terminal. Supports move validation, turn selection, and ASCII board display.

## M1_PG_trained.h5

Current best policy gradient-trained model (M1) after training against M2. Used for evaluation and gameplay.

## DQN_trained.h5

Current best DQN agent trained against M2.

## README.md

Project summary and usage instructions. Customize this file to include model credits, setup steps, and contribution notes.

## Models and Ownership

M1.h5 - Baseline model (before training) — Eshaan

M1_PG_trained.h5 - Policy gradient–trained version of M1 — Eshaan

DQN_trained.h5 - DQN-trained model by Eshaan — Eshaan

M2.h5 - Baseline model — Nico

M2_PG_trained.h5 - Nico’s PG-trained version of M2 — Nico

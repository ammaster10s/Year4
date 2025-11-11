# coding:utf-8
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import csv
from collections import OrderedDict

# --- Environment and Agent Configuration ---
HEIGHT, WIDTH = 8, 8
NUM_OF_STATES = HEIGHT * WIDTH
NUM_OF_ACTION = 4
MAX_OF_TRIAL = 512
MAIN_LOOP_MAX = 1000
REWARD = 10
initial_node = 9
terminal_node = 54
next_s_value = np.array([-WIDTH, WIDTH, -1, 1])  # [UP, DOWN, LEFT, RIGHT]
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3

# BUG FIX: Added wall cells to create a maze.
# wallcells = [18, 19, 20, 21, 22, 29, 37, 45, 53]
wallcells = []

class Env:
    def step(self, current_State, action):
        next_state = current_State + next_s_value[action]
        return next_state

    def admissible_action_check(self, state, action):
        if (state < WIDTH and action == UP) or \
           (state >= WIDTH * (HEIGHT - 1) and action == DOWN) or \
           (state % WIDTH == 0 and action == LEFT) or \
           (state % WIDTH == WIDTH - 1) and action == RIGHT:
            return False
        return True

    def get_reward(self, next_state):
        done = False
        reward = 0  # Small negative reward for each step to encourage shorter paths
        if next_state in wallcells:
            reward = -10
            # Note: We don't set done=True here, agent just gets a penalty.
        elif next_state == terminal_node:
            reward = 100
            done = True
        return reward, done

class Agent:
    def __init__(self, alpha, gamma, epsilon):
        self.env = Env()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def select_action(self, state, qvalue):
        # Get all admissible actions from the current state
        admissible_actions = [a for a in range(NUM_OF_ACTION) if self.env.admissible_action_check(state, a)]

        if not admissible_actions:
            return None # Should not happen in this grid world

        if random.random() < self.epsilon:
            # Explore: choose a random admissible action
            return random.choice(admissible_actions)
        else:
            # Exploit: choose the best admissible action from Q-table
            # Filter q-values for admissible actions only
            q_admissible = {a: qvalue[state][a] for a in admissible_actions}
            return max(q_admissible, key=q_admissible.get)


    def update_Q(self, s, snext, a, qvalue, done, reward):
        if done:
            qvalue[s][a] += self.alpha * (reward - qvalue[s][a])
        else:
            qvalue[s][a] += self.alpha * (reward + self.gamma * qvalue[snext].max() - qvalue[s][a])
        return qvalue

def run_experiment(alpha, gamma, epsilon):
    env = Env()
    agent = Agent(alpha, gamma, epsilon)
    # Initialize Q-table with zeros for better predictability
    q_value = np.zeros((NUM_OF_STATES, NUM_OF_ACTION))
    data = []

    for ep in range(MAIN_LOOP_MAX):
        current_state = initial_node
        done = False
        step = 0
        for _ in range(MAX_OF_TRIAL):
            action = agent.select_action(current_state, q_value)
            if action is None: break

            next_state_candidate = env.step(current_state, action)
            reward, done = env.get_reward(next_state_candidate)

            # IMPROVEMENT: Agent doesn't move if it hits a wall.
            if next_state_candidate in wallcells:
                # The agent attempted to move into a wall.
                # Update Q-value but don't change the state.
                q_value = agent.update_Q(current_state, next_state_candidate, action, q_value, False, reward)
                # We can add a step penalty without moving.
            else:
                # The move is valid, update Q-value and move to the next state.
                q_value = agent.update_Q(current_state, next_state_candidate, action, q_value, done, reward)
                current_state = next_state_candidate

            step += 1
            if done:
                break
        data.append(step)
    return np.array(data)

# --- Visualization Functions ---

def plot_detailed_learning_curves(results, epsilons, alphas, gammas):
    """
    Creates a grid of plots to show learning curves without clutter.
    """
    fig, axes = plt.subplots(len(alphas), len(epsilons), figsize=(20, 12), sharey=True)
    fig.suptitle('Detailed Learning Curves by α and ε', fontsize=16)

    for i, a in enumerate(alphas):
        for j, eps in enumerate(epsilons):
            ax = axes[i, j]
            for g in gammas:
                label = f"γ={g}"
                steps = results[(a, g, eps)]
                # Smooth the curve for better readability using a moving average
                smooth_steps = np.convolve(steps, np.ones(10)/10, mode='valid')
                ax.plot(smooth_steps, label=label)
            ax.set_title(f"α={a}, ε={eps}")
            ax.grid(True)
            ax.legend()

    for ax in axes[-1, :]:
        ax.set_xlabel("Episode")
    for ax in axes[:, 0]:
        ax.set_ylabel("Steps to Goal (Smoothed)")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_summary_performance(results):
    """
    Calculates the average performance over the last part of the training
    and plots it as a sorted bar chart to easily find the best setting.
    """
    # Use OrderedDict to maintain a sorted list of results
    performance = OrderedDict()

    for params, steps in results.items():
        # Calculate average steps over the last 20% of episodes
        average_final_steps = np.mean(steps[-int(MAIN_LOOP_MAX * 0.2):])
        performance[params] = average_final_steps

    # Sort the results by performance (lower is better)
    sorted_performance = sorted(performance.items(), key=lambda x: x[1])

    # Prepare data for plotting
    labels = [f"α={k[0]}, γ={k[1]}, ε={k[2]}" for k, v in sorted_performance]
    values = [v for k, v in sorted_performance]

    plt.figure(figsize=(15, 10))
    bars = plt.barh(labels, values)
    plt.xlabel('Average Steps to Goal (Final 20% of Episodes)')
    plt.title('Q-Learning Hyperparameter Performance Comparison')
    plt.gca().invert_yaxis() # Display best (lowest) at the top

    # Add value labels to each bar
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.2f}',
                 va='center', ha='left')

    plt.tight_layout()
    plt.show()


def main():
    alphas = [0.1, 0.2, 0.5]
    gammas = [0.3, 0.5, 0.9]
    epsilons = [0.1, 0.3, 0.9]

    results = {}

    for eps in epsilons:
        for a in alphas:
            for g in gammas:
                print(f"Running α={a}, γ={g}, ε={eps} ...")
                steps = run_experiment(a, g, eps)
                results[(a, g, eps)] = steps

    # --- Plot Results ---
    # METHOD 1: Plot detailed learning curves in a grid
    plot_detailed_learning_curves(results, epsilons, alphas, gammas)

    # METHOD 2: Plot a summary bar chart to find the best setting
    plot_summary_performance(results)


if __name__ == "__main__":
    main()
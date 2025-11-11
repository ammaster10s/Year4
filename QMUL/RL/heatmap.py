# coding:utf-8
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- Environment and Agent Configuration ---
HEIGHT, WIDTH = 8, 8
NUM_OF_STATES = HEIGHT * WIDTH
NUM_OF_ACTION = 4
MAX_OF_TRIAL = 512
MAIN_LOOP_MAX = 1000  # Number of episodes for training
REWARD = 10
initial_node = 9
terminal_node = 54
next_s_value = np.array([-WIDTH, WIDTH, -1, 1])  # [UP, DOWN, LEFT, RIGHT]
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3

# Define some walls to make the problem more interesting
wallcells = [18, 19, 20, 21, 22, 29, 37, 45, 53]

class Env:
    def step(self, current_State, action):
        return current_State + next_s_value[action]

    def admissible_action_check(self, state, action):
        if (state < WIDTH and action == UP) or \
           (state >= WIDTH * (HEIGHT - 1) and action == DOWN) or \
           (state % WIDTH == 0 and action == LEFT) or \
           (state % WIDTH == WIDTH - 1 and action == RIGHT):
            return False
        return True

    def get_reward(self, next_state):
        done = False
        reward = -0.1  # Small negative reward for each step to encourage shorter paths
        if next_state in wallcells:
            reward = -10
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
        admissible_actions = [a for a in range(NUM_OF_ACTION) if self.env.admissible_action_check(state, a)]
        if not admissible_actions: return None
        if random.random() < self.epsilon:
            return random.choice(admissible_actions)
        else:
            q_admissible = {a: qvalue[state][a] for a in admissible_actions}
            return max(q_admissible, key=q_admissible.get)

    def update_Q(self, s, snext, a, qvalue, done, reward):
        if done:
            qvalue[s][a] += self.alpha * (reward - qvalue[s][a])
        else:
            qvalue[s][a] += self.alpha * (reward + self.gamma * qvalue[snext].max() - qvalue[s][a])
        return qvalue

def run_experiment_and_get_q_table(alpha, gamma, initial_epsilon):
    """
    Runs the Q-learning experiment and returns the final learned Q-table.
    """
    env = Env()
    agent = Agent(alpha, gamma, initial_epsilon)
    q_value = np.zeros((NUM_OF_STATES, NUM_OF_ACTION))
    
    epsilon = initial_epsilon
    min_epsilon = 0.01
    epsilon_decay_rate = 1 / MAIN_LOOP_MAX 

    for ep in range(MAIN_LOOP_MAX):
        current_state = initial_node
        done = False
        agent.epsilon = epsilon

        for _ in range(MAX_OF_TRIAL):
            action = agent.select_action(current_state, q_value)
            if action is None: break
            
            next_state_candidate = env.step(current_state, action)
            reward, done = env.get_reward(next_state_candidate)
            
            if next_state_candidate in wallcells:
                q_value = agent.update_Q(current_state, next_state_candidate, action, q_value, False, reward)
            else:
                q_value = agent.update_Q(current_state, next_state_candidate, action, q_value, done, reward)
                current_state = next_state_candidate
            
            if done: break
        
        epsilon = max(min_epsilon, epsilon - epsilon_decay_rate)
            
    return q_value

def extract_final_path(q_table, start_node):
    """
    Extracts the learned path by greedily following the highest Q-values.
    """
    env = Env()
    path = [start_node]
    current_state = start_node
    
    for _ in range(MAX_OF_TRIAL): # Safety break
        if current_state == terminal_node:
            break
            
        admissible_actions = [a for a in range(NUM_OF_ACTION) if env.admissible_action_check(current_state, a)]
        if not admissible_actions: break 
            
        q_admissible = {a: q_table[current_state][a] for a in admissible_actions}
        best_action = max(q_admissible, key=q_admissible.get)
        
        next_state = env.step(current_state, best_action)
        # Avoid getting stuck in a two-state loop
        if len(path) > 2 and next_state == path[-2]:
            break
        path.append(next_state)
        current_state = next_state
        
    return path

def plot_routes_and_heatmaps(q_tables, alphas, gammas, epsilons):
    """
    Plots a grid of Q-value heatmaps and the final learned routes.
    """
    fig, axes = plt.subplots(len(alphas) * len(gammas), len(epsilons), figsize=(12, 24))
    fig.suptitle('Final Policy and Q-Value Heatmap for Each Setting', fontsize=20)

    # To ensure consistent coloring, find the global min and max Q-values
    all_values = np.concatenate([np.max(qt, axis=1) for qt in q_tables.values()])
    vmin, vmax = np.min(all_values), np.max(all_values)

    im = None
    for j, eps in enumerate(epsilons):
        for i, a in enumerate(alphas):
            for k, g in enumerate(gammas):
                row_idx = i * len(gammas) + k
                ax = axes[row_idx, j]
                params = (a, g, eps)
                q_table = q_tables[params]
                
                # --- Create the Heatmap ---
                # The value of a state is the max Q-value of any action from that state
                value_function = np.max(q_table, axis=1)
                heatmap_data = value_function.reshape((HEIGHT, WIDTH))
                im = ax.imshow(heatmap_data, cmap='viridis', vmin=vmin, vmax=vmax)

                # --- Overlay Grid and Special Cells ---
                ax.set_xticks(np.arange(-0.5, WIDTH, 1), minor=True)
                ax.set_yticks(np.arange(-0.5, HEIGHT, 1), minor=True)
                ax.grid(which="minor", color="white", linestyle='-', linewidth=0.5)
                ax.set_aspect('equal')
                ax.set_title(f"α={a}, γ={g}, ε={eps}", fontsize=10)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                
                start_row, start_col = divmod(initial_node, WIDTH)
                goal_row, goal_col = divmod(terminal_node, WIDTH)
                # Use edgecolor to make start/goal visible
                ax.add_patch(patches.Rectangle((start_col - 0.5, start_row - 0.5), 1, 1, facecolor='lime', edgecolor='white', lw=2))
                ax.add_patch(patches.Rectangle((goal_col - 0.5, goal_row - 0.5), 1, 1, facecolor='red', edgecolor='white', lw=2))
                for wall in wallcells:
                    wall_row, wall_col = divmod(wall, WIDTH)
                    ax.add_patch(patches.Rectangle((wall_col - 0.5, wall_row - 0.5), 1, 1, facecolor='black'))

                # --- Overlay the Final Path ---
                path = extract_final_path(q_table, initial_node)
                for step in range(len(path) - 1):
                    start_pos, end_pos = path[step], path[step+1]
                    start_r, start_c = divmod(start_pos, WIDTH)
                    end_r, end_c = divmod(end_pos, WIDTH)
                    # Use a bright color for the path that stands out
                    ax.arrow(start_c, start_r, end_c - start_c, end_r - start_r, 
                             head_width=0.3, length_includes_head=True, color='magenta', zorder=10)

    # Add a single color bar for the entire figure
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label='State Value (Max Q-Value)')
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()

def main():
    alphas = [0.1, 0.5]  # Reduced for faster demonstration
    gammas = [0.5, 0.9]
    epsilons = [0.1, 0.9] # Reduced to show a clear failure vs success case

    final_q_tables = {}

    for eps in epsilons:
        for a in alphas:
            for g in gammas:
                print(f"Running α={a}, γ={g}, ε={eps} ...")
                final_q_table = run_experiment_and_get_q_table(a, g, eps)
                final_q_tables[(a, g, eps)] = final_q_table

    plot_routes_and_heatmaps(final_q_tables, alphas=alphas, gammas=gammas, epsilons=epsilons)

if __name__ == "__main__":
    main()
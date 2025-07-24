"""
Stochastic Maze Navigation using Value Iteration
-----------------------------------------------
Implements an MDP solution for maze navigation with probabilistic movements.
The agent discovers obstacles dynamically and replans using value iteration.
"""

import random
import time
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# ======================
# GLOBAL CONFIGURATION
# ======================
ROWS, COLS = 5, 6
START_STATE = (0, 0)
GOAL_STATE = (4, 5)
TRUE_OBSTACLES = {(0, 1), (2, 1), (2, 3), (3, 1), (3, 4), (4, 4)}
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
ACTION_PROBS = {
    'UP':    {'UP': 0.8, 'LEFT': 0.1, 'RIGHT': 0.1},
    'DOWN':  {'DOWN': 0.8, 'LEFT': 0.1, 'RIGHT': 0.1},
    'LEFT':  {'LEFT': 0.8, 'UP': 0.1, 'DOWN': 0.1},
    'RIGHT': {'RIGHT': 0.8, 'UP': 0.1, 'DOWN': 0.1},
}
VISUALIZE_EACH_STEP = True  # Set to False for faster execution

# ======================
# MAZE NAVIGATION CLASS
# ======================
class MazeNavigator:
    """Implements probabilistic maze navigation with dynamic obstacle discovery"""
    
    def __init__(self):
        self.known_obstacles = set()
        self.path_history = []
        self.value_history = []
        self.policy_history = []
        
    def is_out_of_bounds(self, state):
        r, c = state
        return r < 0 or r >= ROWS or c < 0 or c >= COLS

    def is_real_obstacle(self, state):
        return state in TRUE_OBSTACLES

    def get_next_state(self, state, action):
        r, c = state
        direction_map = {
            'UP': (r - 1, c),
            'DOWN': (r + 1, c),
            'LEFT': (r, c - 1),
            'RIGHT': (r, c + 1)
        }
        return direction_map[action]

    def get_all_states(self):
        return [(r, c) for r in range(ROWS) for c in range(COLS) 
                if (r, c) not in self.known_obstacles]

    # ======================
    # MDP SOLVING METHODS
    # ======================
    def value_iteration(self, gamma=0.9, epsilon=0.01):
        """Performs value iteration to solve the MDP"""
        states = self.get_all_states()
        V = {s: 0 for s in states}
        
        while True:
            delta = 0
            new_V = {}
            for state in states:
                if state == GOAL_STATE:
                    new_V[state] = 0
                    continue
                
                best_value = float('-inf')
                for action in ACTIONS:
                    exp_utility = self._calculate_expected_utility(state, action, V, gamma)
                    best_value = max(best_value, exp_utility)
                
                new_V[state] = best_value
                delta = max(delta, abs(new_V[state] - V.get(state, 0)))
            
            V = new_V
            if delta < epsilon:
                break
                
        return V

    def _calculate_expected_utility(self, state, action, V, gamma):
        """Helper for value iteration: calculates expected utility of an action"""
        exp_utility = 0
        for a_prime, prob in ACTION_PROBS[action].items():
            next_state = self.get_next_state(state, a_prime)
            
            # Handle invalid states
            if self.is_out_of_bounds(next_state) or next_state in self.known_obstacles:
                reward = -10
                next_state = state  # Stay in current state
            elif next_state == GOAL_STATE:
                reward = 100
            else:
                reward = -1
                
            exp_utility += prob * (reward + gamma * V.get(next_state, 0))
        return exp_utility

    def extract_policy(self, V):
        """Derives optimal policy from value function"""
        states = self.get_all_states()
        policy = {}
        
        for state in states:
            if state == GOAL_STATE:
                policy[state] = None
                continue
                
            best_action, best_value = None, float('-inf')
            for action in ACTIONS:
                exp_utility = self._calculate_expected_utility(state, action, V, gamma=0.9)
                if exp_utility > best_value:
                    best_value = exp_utility
                    best_action = action
                    
            policy[state] = best_action
            
        return policy

    # ======================
    # SIMULATION METHODS
    # ======================
    def execute_action(self, state, action):
        """Simulates action execution with probabilistic outcomes"""
        rand_val = random.random()
        cumulative = 0
        
        for a_prime, prob in ACTION_PROBS[action].items():
            cumulative += prob
            if rand_val < cumulative:
                next_state = self.get_next_state(state, a_prime)
                break
                
        # Handle collisions
        if self.is_out_of_bounds(next_state) or self.is_real_obstacle(next_state):
            self.known_obstacles.add(next_state)
            return state, -10
        return next_state, 100 if next_state == GOAL_STATE else -1

    def navigate(self):
        """Main navigation loop with dynamic replanning"""
        current_state = START_STATE
        self.path_history = [current_state]
        step = 1
        
        while current_state != GOAL_STATE:
            print(f"\n=== Step {step} ===")
            
            # Solve MDP
            V = self.value_iteration()
            policy = self.extract_policy(V)
            
            # Record history for visualization
            self.value_history.append(V.copy())
            self.policy_history.append(policy.copy())
            
            # Execute action
            if current_state not in policy or policy[current_state] is None:
                print("No valid policy from current state!")
                break
                
            action = policy[current_state]
            next_state, reward = self.execute_action(current_state, action)
            
            print(f"State: {current_state}, Action: {action}, " 
                  f"Next: {next_state}, Reward: {reward}")
            
            # Update state
            current_state = next_state
            self.path_history.append(current_state)
            
            # Visualize
            if VISUALIZE_EACH_STEP:
                self.visualize(current_state)
                time.sleep(0.5)
                
            step += 1
            
        # Final output
        print("\n=== Navigation Complete ===")
        print(f"Final Path: {self.path_history}")
        print(f"Total Steps: {len(self.path_history)-1}")
        self.visualize_final()

    # ======================
    # VISUALIZATION METHODS
    # ======================
    def visualize(self, agent_pos):
        """Dynamic visualization of current maze state"""
        grid = np.zeros((ROWS, COLS))
        
        # Create grid representation
        for r in range(ROWS):
            for c in range(COLS):
                if (r, c) in self.known_obstacles:
                    grid[r, c] = -1
                elif (r, c) == START_STATE:
                    grid[r, c] = 2
                elif (r, c) == GOAL_STATE:
                    grid[r, c] = 3
                elif (r, c) == agent_pos:
                    grid[r, c] = 4
                else:
                    grid[r, c] = 0

        # Create colormap
        cmap = ListedColormap(['lightgray', 'darkred', 'royalblue', 'limegreen', 'gold'])
        norm = plt.Normalize(vmin=-1, vmax=4)
        
        # Plot grid
        plt.figure(figsize=(8, 6))
        plt.imshow(grid, cmap=cmap, norm=norm, interpolation='nearest')
        
        # Add annotations
        for r in range(ROWS):
            for c in range(COLS):
                if (r, c) in self.known_obstacles:
                    plt.text(c, r, 'X', ha='center', va='center', 
                            color='white', fontweight='bold', fontsize=14)
                elif (r, c) == START_STATE:
                    plt.text(c, r, 'START', ha='center', va='center', 
                            color='white', fontweight='bold')
                elif (r, c) == GOAL_STATE:
                    plt.text(c, r, 'GOAL', ha='center', va='center', 
                            color='white', fontweight='bold')
                elif grid[r, c] == 4:  # Agent position
                    plt.text(c, r, 'AGENT', ha='center', va='center', 
                            color='black', fontweight='bold')
                elif (r, c) in self.path_history:
                    idx = self.path_history.index((r, c))
                    plt.text(c, r, str(idx), ha='center', va='center', 
                            color='purple', fontweight='bold')
        
        plt.title(f"Maze Navigation (Step {len(self.path_history)-1})")
        plt.xticks(range(COLS))
        plt.yticks(range(ROWS))
        plt.grid(color='white', linestyle='-', linewidth=1)
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)
        plt.close()

    def visualize_final(self):
        """Comprehensive visualization of entire path"""
        plt.figure(figsize=(10, 8))
        
        # Create path visualization
        path_arr = np.array(self.path_history)
        plt.plot(path_arr[:,1], path_arr[:,0], 'o-', 
                linewidth=3, markersize=12, color='purple', 
                markerfacecolor='orchid', label='Agent Path')
        
        # Mark special points
        plt.plot(START_STATE[1], START_STATE[0], 's', markersize=15, 
                color='royalblue', label='Start', markerfacecolor='lightblue')
        plt.plot(GOAL_STATE[1], GOAL_STATE[0], 'D', markersize=15, 
                color='limegreen', label='Goal', markerfacecolor='palegreen')
        
        # Add obstacles
        obs_arr = np.array(list(TRUE_OBSTACLES))
        plt.scatter(obs_arr[:,1], obs_arr[:,0], s=800, marker='X', 
                   color='darkred', label='Obstacles')
        
        # Configure plot
        plt.xlim(-0.5, COLS-0.5)
        plt.ylim(ROWS-0.5, -0.5)  # Invert y-axis for matrix coordinates
        plt.xticks(range(COLS))
        plt.yticks(range(ROWS))
        plt.grid(True, color='gray', linestyle='--')
        plt.title("Maze Navigation Path", fontsize=16)
        plt.xlabel("Column", fontsize=12)
        plt.ylabel("Row", fontsize=12)
        plt.legend(loc='upper right')
        
        # Add annotations
        for i, (r, c) in enumerate(self.path_history):
            plt.text(c, r, str(i), ha='center', va='center', 
                    color='white', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('maze_path.png', dpi=300)
        plt.show()

# ======================
# MAIN EXECUTION
# ======================
if __name__ == "__main__":
    print("Starting Stochastic Maze Navigation...")
    print(f"Maze Size: {ROWS}x{COLS}")
    print(f"Start: {START_STATE}, Goal: {GOAL_STATE}")
    print(f"True Obstacles: {TRUE_OBSTACLES}")
    
    navigator = MazeNavigator()
    start_time = time.time()
    navigator.navigate()
    
    duration = time.time() - start_time
    print(f"\nExecution Time: {duration:.2f} seconds")
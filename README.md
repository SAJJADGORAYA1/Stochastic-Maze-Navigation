# Stochastic Maze Navigation with Value Iteration

## Overview
Implementation of a Markov Decision Process (MDP) solution for probabilistic maze navigation. The agent:
- Discovers obstacles dynamically during exploration
- Uses value iteration for optimal path planning
- Handles stochastic movement uncertainties
- Visualizes the navigation path in real-time

## Key Features
- **Dynamic Obstacle Discovery**: Agents learn obstacles through interaction
- **Probabilistic Transitions**: 80% chance of intended move, 20% chance of perpendicular moves
- **Replanning**: Updates policy after each step with new obstacle information
- **Visualization**: Real-time path tracking and final path visualization

## Technical Components
1. **MDP Formulation**:
   - States: Grid positions
   - Actions: {UP, DOWN, LEFT, RIGHT}
   - Rewards: +100 (goal), -10 (obstacle), -1 (step)
   - Discount Factor: γ = 0.9

2. **Value Iteration**:
   ```python
   while delta > epsilon:
       for state in states:
           Q_values = []
           for action in actions:
               expected_utility = calculate_expected_utility(state, action)
               Q_values.append(expected_utility)
           V[state] = max(Q_values)
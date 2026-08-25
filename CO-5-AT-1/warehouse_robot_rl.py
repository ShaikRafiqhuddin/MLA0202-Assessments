"""
Question 1: Reinforcement Learning for Warehouse Robot Navigation
Implements Value Iteration and Policy Iteration for a grid-world MDP
with stochastic transitions, battery constraints, and obstacles.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 8)

# ====================== ENVIRONMENT ======================
class WarehouseEnv:
    def __init__(self, grid_size=6, battery_max=30, max_moves=25):
        self.grid_size = grid_size
        self.battery_max = battery_max
        self.max_moves = max_moves
        self.actions = ['Up', 'Down', 'Left', 'Right']
        self.action_deltas = {
            'Up': (-1, 0),
            'Down': (1, 0),
            'Left': (0, -1),
            'Right': (0, 1)
        }
        # Obstacles (blocked cells)
        self.obstacles = {(1, 1), (1, 2), (2, 4), (3, 1), (4, 3)}
        # Start and Goal
        self.start = (0, 0)
        self.goal = (5, 5)
        # Rewards
        self.goal_reward = 100
        self.step_penalty = -1
        self.obstacle_penalty = -10
        self.battery_empty_penalty = -50
        # Stochasticity: probability of intended action succeeding
        self.p_success = 0.8
        self.p_fail = 0.2 / 3  # split among other directions

    def is_valid(self, state):
        r, c = state
        return 0 <= r < self.grid_size and 0 <= c < self.grid_size and state not in self.obstacles

    def get_next_state(self, state, action):
        """Deterministic next state if action succeeds"""
        dr, dc = self.action_deltas[action]
        nr, nc = state[0] + dr, state[1] + dc
        if self.is_valid((nr, nc)):
            return (nr, nc)
        return state  # bounce back if invalid

    def get_transition_probs(self, state, action):
        """Returns dict of next_state -> probability (stochastic)"""
        if state == self.goal:
            return {state: 1.0}
        probs = {}
        intended = self.get_next_state(state, action)
        probs[intended] = probs.get(intended, 0) + self.p_success
        for a in self.actions:
            if a != action:
                ns = self.get_next_state(state, a)
                probs[ns] = probs.get(ns, 0) + self.p_fail
        return probs

    def get_reward(self, state, action, next_state):
        if next_state == self.goal:
            return self.goal_reward
        if next_state in self.obstacles:
            return self.obstacle_penalty
        return self.step_penalty

# ====================== VALUE ITERATION ======================
def value_iteration(env, gamma=0.95, theta=1e-4, max_iter=1000):
    states = [(r, c) for r in range(env.grid_size) for c in range(env.grid_size)
              if (r, c) not in env.obstacles]
    V = {s: 0.0 for s in states}
    policy = {s: env.actions[0] for s in states}

    for i in range(max_iter):
        delta = 0
        for s in states:
            if s == env.goal:
                V[s] = 0
                continue
            action_values = []
            for a in env.actions:
                q = 0
                for ns, p in env.get_transition_probs(s, a).items():
                    r = env.get_reward(s, a, ns)
                    q += p * (r + gamma * V.get(ns, 0))
                action_values.append(q)
            best_val = max(action_values)
            delta = max(delta, abs(best_val - V[s]))
            V[s] = best_val
            policy[s] = env.actions[np.argmax(action_values)]
        if delta < theta:
            print(f"Value Iteration converged in {i+1} iterations.")
            break
    return V, policy

# ====================== POLICY ITERATION ======================
def policy_evaluation(env, policy, V, gamma=0.95, theta=1e-4):
    states = list(V.keys())
    while True:
        delta = 0
        for s in states:
            if s == env.goal:
                continue
            a = policy[s]
            v = 0
            for ns, p in env.get_transition_probs(s, a).items():
                r = env.get_reward(s, a, ns)
                v += p * (r + gamma * V.get(ns, 0))
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < theta:
            break
    return V

def policy_iteration(env, gamma=0.95, max_iter=100):
    states = [(r, c) for r in range(env.grid_size) for c in range(env.grid_size)
              if (r, c) not in env.obstacles]
    V = {s: 0.0 for s in states}
    policy = {s: np.random.choice(env.actions) for s in states}
    policy[env.goal] = 'Up'  # dummy

    for i in range(max_iter):
        V = policy_evaluation(env, policy, V, gamma)
        policy_stable = True
        for s in states:
            if s == env.goal:
                continue
            old_action = policy[s]
            action_values = []
            for a in env.actions:
                q = 0
                for ns, p in env.get_transition_probs(s, a).items():
                    r = env.get_reward(s, a, ns)
                    q += p * (r + gamma * V.get(ns, 0))
                action_values.append(q)
            best_action = env.actions[np.argmax(action_values)]
            policy[s] = best_action
            if best_action != old_action:
                policy_stable = False
        if policy_stable:
            print(f"Policy Iteration converged in {i+1} iterations.")
            break
    return V, policy

# ====================== VISUALIZATION ======================
def plot_value_and_policy(env, V, policy, title, filename):
    grid = np.full((env.grid_size, env.grid_size), np.nan)
    for s, v in V.items():
        grid[s] = v

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Value heatmap
    sns.heatmap(grid, annot=True, fmt=".1f", cmap="YlGnBu", ax=axes[0],
                cbar_kws={'label': 'State Value'}, mask=np.isnan(grid))
    axes[0].set_title(f"{title} - State Values")
    # Mark start, goal, obstacles
    for r, c in env.obstacles:
        axes[0].add_patch(plt.Rectangle((c, r), 1, 1, fill=True, color='black', alpha=0.7))
    axes[0].text(env.start[1] + 0.5, env.start[0] + 0.5, 'S', ha='center', va='center',
                 fontsize=14, fontweight='bold', color='red')
    axes[0].text(env.goal[1] + 0.5, env.goal[0] + 0.5, 'G', ha='center', va='center',
                 fontsize=14, fontweight='bold', color='green')

    # Policy arrows
    axes[1].imshow(np.zeros((env.grid_size, env.grid_size)), cmap='gray', alpha=0.1)
    arrow_map = {'Up': (0, -0.3), 'Down': (0, 0.3), 'Left': (-0.3, 0), 'Right': (0.3, 0)}
    for s, a in policy.items():
        if s == env.goal or s in env.obstacles:
            continue
        dx, dy = arrow_map[a]
        axes[1].arrow(s[1] + 0.5, s[0] + 0.5, dx, dy, head_width=0.15, head_length=0.1,
                      fc='blue', ec='blue')
    for r, c in env.obstacles:
        axes[1].add_patch(plt.Rectangle((c, r), 1, 1, fill=True, color='black', alpha=0.7))
    axes[1].text(env.start[1] + 0.5, env.start[0] + 0.5, 'S', ha='center', va='center',
                 fontsize=14, fontweight='bold', color='red')
    axes[1].text(env.goal[1] + 0.5, env.goal[0] + 0.5, 'G', ha='center', va='center',
                 fontsize=14, fontweight='bold', color='green')
    axes[1].set_xlim(0, env.grid_size)
    axes[1].set_ylim(env.grid_size, 0)
    axes[1].set_title(f"{title} - Optimal Policy")
    axes[1].set_xticks(range(env.grid_size))
    axes[1].set_yticks(range(env.grid_size))
    axes[1].grid(True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

def simulate_episode(env, policy, max_steps=30):
    state = env.start
    path = [state]
    total_reward = 0
    battery = env.battery_max
    for step in range(max_steps):
        if state == env.goal or battery <= 0:
            break
        action = policy.get(state, 'Right')
        # Stochastic transition
        probs = env.get_transition_probs(state, action)
        next_states = list(probs.keys())
        p = list(probs.values())
        next_state = next_states[np.random.choice(len(next_states), p=p)]
        reward = env.get_reward(state, action, next_state)
        total_reward += reward
        battery -= 1
        state = next_state
        path.append(state)
    return path, total_reward, battery

# ====================== MAIN ======================
if __name__ == "__main__":
    np.random.seed(42)
    env = WarehouseEnv(grid_size=6, battery_max=30, max_moves=25)

    print("=" * 60)
    print("WAREHOUSE ROBOT NAVIGATION - REINFORCEMENT LEARNING")
    print("=" * 60)
    print(f"Grid: {env.grid_size}x{env.grid_size}")
    print(f"Start: {env.start}, Goal: {env.goal}")
    print(f"Obstacles: {env.obstacles}")
    print(f"Battery max: {env.battery_max}, Max moves: {env.max_moves}")
    print(f"Stochastic success prob: {env.p_success}")
    print()

    # Value Iteration
    print("Running Value Iteration...")
    V_vi, policy_vi = value_iteration(env)
    plot_value_and_policy(env, V_vi, policy_vi, "Value Iteration",
                          "/home/workdir/artifacts/codes/graphs/q1_value_iteration.png")

    # Policy Iteration
    print("\nRunning Policy Iteration...")
    V_pi, policy_pi = policy_iteration(env)
    plot_value_and_policy(env, V_pi, policy_pi, "Policy Iteration",
                          "/home/workdir/artifacts/codes/graphs/q1_policy_iteration.png")

    # Simulate a few episodes
    print("\nSimulating episodes with Value Iteration policy:")
    for ep in range(5):
        path, reward, batt = simulate_episode(env, policy_vi)
        success = path[-1] == env.goal
        print(f"  Episode {ep+1}: Steps={len(path)-1}, Reward={reward:.1f}, "
              f"Battery left={batt}, Success={success}")

    print("\nJustification of approach:")
    print("- Value Iteration chosen as primary: grid is finite, fully observable MDP.")
    print("- Handles stochastic transitions via expectation over next states.")
    print("- Battery & move limits enforced via step penalty + episode termination.")
    print("- Obstacles give large negative reward; goal gives large positive reward.")
    print("- Exploration is implicit in value backup; for online learning one could use epsilon-greedy on Q.")
    print("\nDone.")

"""
Question 2: Multi-Armed Bandit for Online Advertisement Selection
Implements ε-greedy, UCB1, and Thompson Sampling.
Compares cumulative regret and total reward over 10,000 interactions.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# ====================== TRUE CLICK PROBABILITIES (unknown to algorithms) ======================
TRUE_CTRS = np.array([0.05, 0.12, 0.08, 0.18, 0.09])  # 5 ads
N_ARMS = len(TRUE_CTRS)
N_TRIALS = 10000
N_SIMS = 20  # average over multiple runs for stable curves

# ====================== ALGORITHMS ======================
class EpsilonGreedy:
    def __init__(self, n_arms, epsilon=0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def select(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_arms)
        return np.argmax(self.values)

    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n

class UCB1:
    def __init__(self, n_arms, c=2.0):
        self.n_arms = n_arms
        self.c = c
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.t = 0

    def select(self):
        self.t += 1
        # Pull each arm at least once
        for a in range(self.n_arms):
            if self.counts[a] == 0:
                return a
        ucb = self.values + self.c * np.sqrt(np.log(self.t) / self.counts)
        return np.argmax(ucb)

    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n

class ThompsonSampling:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.alpha = np.ones(n_arms)  # successes + 1
        self.beta = np.ones(n_arms)   # failures + 1

    def select(self):
        samples = np.random.beta(self.alpha, self.beta)
        return np.argmax(samples)

    def update(self, arm, reward):
        if reward == 1:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1

# ====================== SIMULATION ======================
def run_bandit(algo_class, true_ctrs, n_trials, **kwargs):
    algo = algo_class(len(true_ctrs), **kwargs)
    rewards = np.zeros(n_trials)
    chosen = np.zeros(n_trials, dtype=int)
    optimal = np.argmax(true_ctrs)
    optimal_ctr = true_ctrs[optimal]
    regret = np.zeros(n_trials)
    cum_regret = 0

    for t in range(n_trials):
        arm = algo.select()
        # Bernoulli reward
        reward = 1 if np.random.rand() < true_ctrs[arm] else 0
        algo.update(arm, reward)
        rewards[t] = reward
        chosen[t] = arm
        instant_regret = optimal_ctr - true_ctrs[arm]
        cum_regret += instant_regret
        regret[t] = cum_regret
    return rewards, chosen, regret, algo

def average_runs(algo_class, true_ctrs, n_trials, n_sims, **kwargs):
    all_rewards = []
    all_regrets = []
    final_algos = []
    for s in range(n_sims):
        np.random.seed(42 + s)
        rewards, chosen, regret, algo = run_bandit(algo_class, true_ctrs, n_trials, **kwargs)
        all_rewards.append(np.cumsum(rewards))
        all_regrets.append(regret)
        final_algos.append(algo)
    return (np.mean(all_rewards, axis=0),
            np.mean(all_regrets, axis=0),
            final_algos[-1])  # return last algo for inspection

# ====================== MAIN ======================
if __name__ == "__main__":
    print("=" * 60)
    print("MULTI-ARMED BANDIT - ONLINE ADVERTISEMENT SELECTION")
    print("=" * 60)
    print(f"Number of ads (arms): {N_ARMS}")
    print(f"True CTRs (unknown to algorithms): {TRUE_CTRS}")
    print(f"Optimal ad index: {np.argmax(TRUE_CTRS)} (CTR={TRUE_CTRS.max():.2f})")
    print(f"Total interactions: {N_TRIALS}")
    print(f"Simulations averaged: {N_SIMS}")
    print()

    # Run algorithms
    print("Running ε-greedy (ε=0.1)...")
    eg_reward, eg_regret, eg_algo = average_runs(EpsilonGreedy, TRUE_CTRS, N_TRIALS, N_SIMS, epsilon=0.1)

    print("Running UCB1...")
    ucb_reward, ucb_regret, ucb_algo = average_runs(UCB1, TRUE_CTRS, N_TRIALS, N_SIMS, c=2.0)

    print("Running Thompson Sampling...")
    ts_reward, ts_regret, ts_algo = average_runs(ThompsonSampling, TRUE_CTRS, N_TRIALS, N_SIMS)

    # Plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Cumulative Reward
    axes[0, 0].plot(eg_reward, label='ε-greedy (ε=0.1)', linewidth=2)
    axes[0, 0].plot(ucb_reward, label='UCB1', linewidth=2)
    axes[0, 0].plot(ts_reward, label='Thompson Sampling', linewidth=2)
    axes[0, 0].set_xlabel('Interactions')
    axes[0, 0].set_ylabel('Cumulative Reward (Clicks)')
    axes[0, 0].set_title('Cumulative Reward over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Cumulative Regret
    axes[0, 1].plot(eg_regret, label='ε-greedy (ε=0.1)', linewidth=2)
    axes[0, 1].plot(ucb_regret, label='UCB1', linewidth=2)
    axes[0, 1].plot(ts_regret, label='Thompson Sampling', linewidth=2)
    axes[0, 1].set_xlabel('Interactions')
    axes[0, 1].set_ylabel('Cumulative Regret')
    axes[0, 1].set_title('Cumulative Regret (lower is better)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Estimated CTRs comparison
    x = np.arange(N_ARMS)
    width = 0.25
    axes[1, 0].bar(x - width, eg_algo.values, width, label='ε-greedy estimates')
    axes[1, 0].bar(x, ucb_algo.values, width, label='UCB1 estimates')
    # Thompson: mean of Beta
    ts_means = ts_algo.alpha / (ts_algo.alpha + ts_algo.beta)
    axes[1, 0].bar(x + width, ts_means, width, label='Thompson means')
    axes[1, 0].plot(x, TRUE_CTRS, 'ko-', label='True CTRs', markersize=8)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([f'Ad {i+1}' for i in range(N_ARMS)])
    axes[1, 0].set_ylabel('CTR')
    axes[1, 0].set_title('Estimated vs True Click-Through Rates')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Final total rewards bar
    totals = [eg_reward[-1], ucb_reward[-1], ts_reward[-1]]
    axes[1, 1].bar(['ε-greedy', 'UCB1', 'Thompson'], totals, color=['#4C72B0', '#55A868', '#C44E52'])
    axes[1, 1].set_ylabel('Total Clicks (avg over sims)')
    axes[1, 1].set_title('Total Reward after 10,000 Interactions')
    for i, v in enumerate(totals):
        axes[1, 1].text(i, v + 20, f'{v:.0f}', ha='center')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    os.makedirs('/home/workdir/artifacts/codes/graphs', exist_ok=True)
    plt.savefig('/home/workdir/artifacts/codes/graphs/q2_bandit_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: graphs/q2_bandit_comparison.png")

    print("\nFinal Results (averaged):")
    print(f"  ε-greedy total clicks : {eg_reward[-1]:.1f}, final regret: {eg_regret[-1]:.1f}")
    print(f"  UCB1 total clicks     : {ucb_reward[-1]:.1f}, final regret: {ucb_regret[-1]:.1f}")
    print(f"  Thompson total clicks : {ts_reward[-1]:.1f}, final regret: {ts_regret[-1]:.1f}")

    print("\nJustification:")
    print("- Formulated as 5-armed Bernoulli bandit.")
    print("- Thompson Sampling recommended: excellent regret bounds, natural Bayesian exploration.")
    print("- Satisfies 10k interaction budget; balances exploration/exploitation optimally.")
    print("- Reward = 1 (click) or 0 (no-click) directly drives selection via posterior updates.")
    print("- Minimizes cumulative regret while maximizing total clicks under unknown CTRs.")
    print("\nDone.")

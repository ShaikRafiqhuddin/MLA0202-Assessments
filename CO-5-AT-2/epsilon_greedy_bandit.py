"""Q2 - Epsilon-Greedy Algorithm for a 5-Armed Movie Bandit."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_epsilon_greedy(n_rounds=10000, epsilon=.10, seed=42):
    rng = np.random.default_rng(seed)
    true_rates = np.array([.12, .20, .08, .15, .28])
    counts = np.zeros(5, dtype=int); estimates = np.zeros(5)
    rows=[]; cumulative=0
    for t in range(1, n_rounds+1):
        if rng.random() < epsilon:
            action=int(rng.integers(5)); mode="exploration"
        else:
            best=np.flatnonzero(np.isclose(estimates, estimates.max()))
            action=int(rng.choice(best)); mode="exploitation"
        reward=int(rng.random() < true_rates[action])
        counts[action]+=1
        estimates[action] += (reward-estimates[action])/counts[action]
        cumulative += reward
        rows.append((t, action+1, mode, reward, estimates[action], cumulative))
    d=pd.DataFrame(rows, columns=["round","movie","mode","reward","estimated_reward","cumulative_reward"])
    d["instant_regret"]=true_rates.max()-d.reward
    d["cumulative_regret"]=d.instant_regret.cumsum()
    return d, true_rates, counts, estimates

def main():
    out=Path(__file__).parent/"outputs"; out.mkdir(exist_ok=True)
    d,rates,counts,est=run_epsilon_greedy()
    d.to_csv(out/"bandit_interactions.csv", index=False)
    s=pd.DataFrame({"movie":[f"Movie_{i}" for i in range(1,6)],
                    "true_click_probability":rates,"selection_count":counts,
                    "estimated_reward":est})
    s.to_csv(out/"movie_summary.csv", index=False)
    plt.figure(figsize=(9,5)); plt.plot(d["round"],d["cumulative_reward"])
    plt.xlabel("Interaction"); plt.ylabel("Cumulative reward"); plt.title("Epsilon-Greedy: Cumulative Engagement")
    plt.tight_layout(); plt.savefig(out/"cumulative_reward.png",dpi=150); plt.close()
    plt.figure(figsize=(9,5)); plt.plot(d["round"],d["cumulative_regret"])
    plt.xlabel("Interaction"); plt.ylabel("Cumulative regret"); plt.title("Epsilon-Greedy: Cumulative Regret")
    plt.tight_layout(); plt.savefig(out/"cumulative_regret.png",dpi=150); plt.close()
    plt.figure(figsize=(8,5)); sns.barplot(data=s,x="movie",y="selection_count",hue="movie",legend=False)
    plt.xlabel("Movie recommendation"); plt.ylabel("Number of selections"); plt.title("Movie Selection Frequency")
    plt.tight_layout(); plt.savefig(out/"selection_frequency.png",dpi=150); plt.close()
    print(f"Movie with highest estimated reward: Movie_{int(np.argmax(est))+1}")
    print(s)

if __name__=="__main__":
    main()

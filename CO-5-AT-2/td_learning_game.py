"""Q4 - Temporal Difference (TD(0)) Learning for Game Score Prediction."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_episode(rng,n_states=8,max_steps=12):
    state=0; ep=[]
    for _ in range(max_steps):
        if state==n_states-1: ep.append((state,0.,state,True)); break
        ns=min(state+int(rng.choice([1,2],p=[.7,.3])),n_states-1)
        reward=1. if ns==n_states-1 else float(rng.normal(0,.1))
        done=ns==n_states-1; ep.append((state,reward,ns,done)); state=ns
        if done:break
    return ep

def td_zero(n_episodes=500,alpha=.10,gamma=.90,seed=42):
    rng=np.random.default_rng(seed); values=np.zeros(8); history=[]
    for e in range(1,n_episodes+1):
        for s,reward,ns,done in generate_episode(rng):
            target=reward if done else reward+gamma*values[ns]
            values[s]+=alpha*(target-values[s])
        history.append((e,float(values[:-1].mean())))
    return values,pd.DataFrame(history,columns=["episode","mean_value"])

def main():
    out=Path(__file__).parent/"outputs"; out.mkdir(exist_ok=True)
    values,h=td_zero()
    pd.DataFrame({"state":np.arange(8),"value":values}).to_csv(out/"state_values.csv",index=False)
    h.to_csv(out/"learning_history.csv",index=False)
    plt.figure(figsize=(9,5)); plt.plot(h.episode,h.mean_value)
    plt.xlabel("Episode");plt.ylabel("Mean value estimate");plt.title("TD(0) Learning Progress")
    plt.tight_layout();plt.savefig(out/"td_learning_progress.png",dpi=150);plt.close()
    print("Final state-value estimates:"); [print(f"State {i}: {v:.4f}") for i,v in enumerate(values)]

if __name__=="__main__":main()

"""Q3 - Value Iteration for Robot Path Planning."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ACTIONS={"UP":(-1,0),"DOWN":(1,0),"LEFT":(0,-1),"RIGHT":(0,1)}

def value_iteration(rows=6,cols=6,start=(5,0),goal=(0,5),obstacles=None,
                    gamma=.90,step_reward=-1.,goal_reward=100.,theta=1e-8):
    obstacles=set(obstacles or {(1,1),(2,1),(3,3),(4,3),(4,4)})
    states=[(r,c) for r in range(rows) for c in range(cols) if (r,c) not in obstacles]
    values={s:0. for s in states}
    def transition(s,a):
        if s==goal:return s,0.
        r,c=s; dr,dc=ACTIONS[a]; ns=(r+dr,c+dc)
        if not(0<=ns[0]<rows and 0<=ns[1]<cols) or ns in obstacles:return s,step_reward
        return ns,(goal_reward if ns==goal else step_reward)
    iterations=0
    while True:
        iterations+=1; delta=0.; new=values.copy()
        for s in states:
            if s==goal:continue
            new[s]=max(reward+gamma*values[ns] for a in ACTIONS for ns,reward in [transition(s,a)])
            delta=max(delta,abs(new[s]-values[s]))
        values=new
        if delta<theta:break
    policy={}
    for s in states:
        if s==goal: policy[s]="GOAL"; continue
        policy[s]=max((transition(s,a)[1]+gamma*values[transition(s,a)[0]],a) for a in ACTIONS)[1]
    return values,policy,iterations

def extract_path(policy,start,goal,max_steps=25):
    s=start; path=[s]; seen={s}
    for _ in range(max_steps):
        if s==goal:break
        dr,dc=ACTIONS[policy[s]]; ns=(s[0]+dr,s[1]+dc)
        if ns in seen:break
        path.append(ns); seen.add(ns); s=ns
    return path

def main():
    out=Path(__file__).parent/"outputs"; out.mkdir(exist_ok=True)
    start,goal=(5,0),(0,5); obstacles={(1,1),(2,1),(3,3),(4,3),(4,4)}
    values,policy,it=value_iteration(start=start,goal=goal,obstacles=obstacles)
    path=extract_path(policy,start,goal)
    pd.DataFrame([{"row":s[0],"col":s[1],"value":v,"policy":policy[s]} for s,v in values.items()]).to_csv(out/"value_policy.csv",index=False)
    pd.DataFrame(path,columns=["row","col"]).assign(step=np.arange(len(path))).to_csv(out/"optimal_path.csv",index=False)
    grid=np.full((6,6),np.nan)
    for s,v in values.items():grid[s]=v
    plt.figure(figsize=(8,6)); plt.imshow(grid,cmap="viridis"); plt.colorbar(label="State value")
    for r in range(6):
        for c in range(6):
            if (r,c) in obstacles:plt.text(c,r,"X",ha="center",va="center",fontsize=16)
            elif (r,c) in policy:plt.text(c,r,"G" if policy[(r,c)]=="GOAL" else policy[(r,c)][0],ha="center",va="center",color="white")
    a=np.array(path); plt.plot(a[:,1],a[:,0],marker="o",linewidth=2)
    plt.scatter([start[1]],[start[0]],s=100,marker="s",label="Start"); plt.scatter([goal[1]],[goal[0]],s=100,marker="*",label="Goal")
    plt.title("Value Iteration: Robot Optimal Policy and Path"); plt.xlabel("Column"); plt.ylabel("Row"); plt.legend()
    plt.tight_layout(); plt.savefig(out/"value_iteration_policy.png",dpi=150); plt.close()
    print(f"Converged in {it} iterations. Path length: {len(path)-1} moves."); print("Path:",path)

if __name__=="__main__":main()

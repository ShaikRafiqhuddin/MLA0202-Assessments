"""Q5 - Policy Iteration for Autonomous Drone Navigation."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ACTIONS={"UP":(-1,0),"DOWN":(1,0),"LEFT":(0,-1),"RIGHT":(0,1)}

def policy_iteration(rows=5,cols=5,start=(4,0),goal=(0,4),obstacles=None,gamma=.90,energy_cost=-1.,goal_reward=50.):
    obstacles=set(obstacles or {(1,1),(2,1),(2,3),(3,3)})
    states=[(r,c) for r in range(rows) for c in range(cols) if (r,c) not in obstacles]
    names=list(ACTIONS); policy={s:names[0] for s in states}; policy[goal]="GOAL"; values={s:0. for s in states}
    def transition(s,a):
        if s==goal:return s,0.
        r,c=s;dr,dc=ACTIONS[a];ns=(r+dr,c+dc)
        if not(0<=ns[0]<rows and 0<=ns[1]<cols) or ns in obstacles:return s,energy_cost
        return ns,(goal_reward if ns==goal else energy_cost)
    rounds=0
    while True:
        for _ in range(1000):
            delta=0.;new=values.copy()
            for s in states:
                if s==goal:continue
                ns,reward=transition(s,policy[s]);new[s]=reward+gamma*values[ns];delta=max(delta,abs(new[s]-values[s]))
            values=new
            if delta<1e-8:break
        stable=True
        for s in states:
            if s==goal:continue
            old=policy[s]
            policy[s]=max((transition(s,a)[1]+gamma*values[transition(s,a)[0]],a) for a in names)[1]
            stable &= policy[s]==old
        rounds+=1
        if stable:break
    return values,policy,rounds

def extract_path(policy,start,goal,max_steps=30):
    s=start;path=[s];seen={s}
    for _ in range(max_steps):
        if s==goal:break
        dr,dc=ACTIONS[policy[s]];ns=(s[0]+dr,s[1]+dc)
        if ns in seen:break
        path.append(ns);seen.add(ns);s=ns
    return path

def main():
    out=Path(__file__).parent/"outputs";out.mkdir(exist_ok=True)
    start,goal=(4,0),(0,4);obstacles={(1,1),(2,1),(2,3),(3,3)}
    values,policy,rounds=policy_iteration(start=start,goal=goal,obstacles=obstacles)
    path=extract_path(policy,start,goal)
    pd.DataFrame([{"row":s[0],"col":s[1],"value":values[s],"policy":policy[s]} for s in values]).to_csv(out/"optimal_policy.csv",index=False)
    pd.DataFrame(path,columns=["row","col"]).assign(step=np.arange(len(path))).to_csv(out/"drone_path.csv",index=False)
    grid=np.full((5,5),np.nan)
    for s,v in values.items():grid[s]=v
    plt.figure(figsize=(7,6));plt.imshow(grid,cmap="plasma");plt.colorbar(label="State value")
    for r in range(5):
        for c in range(5):
            if (r,c) in obstacles:plt.text(c,r,"X",ha="center",va="center",fontsize=16)
            elif (r,c) in policy:plt.text(c,r,"G" if policy[(r,c)]=="GOAL" else policy[(r,c)][0],ha="center",va="center",color="white")
    a=np.array(path);plt.plot(a[:,1],a[:,0],marker="o",linewidth=2)
    plt.scatter([start[1]],[start[0]],s=100,marker="s",label="Start");plt.scatter([goal[1]],[goal[0]],s=100,marker="*",label="Goal")
    plt.title("Policy Iteration: Drone Navigation Policy");plt.xlabel("Column");plt.ylabel("Row");plt.legend()
    plt.tight_layout();plt.savefig(out/"policy_iteration_drone.png",dpi=150);plt.close()
    print(f"Policy stabilized after {rounds} improvement rounds. Path length: {len(path)-1} moves.");print("Path:",path)

if __name__=="__main__":main()

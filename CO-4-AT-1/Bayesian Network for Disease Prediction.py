print("SHAIK RAFIQHUDDIN - 192525129")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
model = DiscreteBayesianNetwork([
    ("Obesity", "Diabetes"),
    ("High_Blood_Sugar", "Diabetes")
])
cpd_obesity = TabularCPD(
    variable="Obesity",
    variable_card=2,
    values=[[0.65], [0.35]]
)
cpd_sugar = TabularCPD(
    variable="High_Blood_Sugar",
    variable_card=2,
    values=[[0.60], [0.40]]
)
cpd_diabetes = TabularCPD(
    variable="Diabetes",
    variable_card=2,
    values=[
        [0.98, 0.80, 0.70, 0.20],
        [0.02, 0.20, 0.30, 0.80]
    ],
    evidence=["Obesity", "High_Blood_Sugar"],
    evidence_card=[2, 2]
)
model.add_cpds(cpd_obesity, cpd_sugar, cpd_diabetes)
print("Model Valid:", model.check_model())
inference = VariableElimination(model)
result = inference.query(
    variables=["Diabetes"],
    evidence={
        "Obesity": 1,
        "High_Blood_Sugar": 1
    }
)
print("\nPredicted Probability of Diabetes:")
print(result)
probability = result.values[1]
print(f"\nProbability of Diabetes: {probability * 100:.2f}%")
graph = nx.DiGraph()
graph.add_edges_from(model.edges())
plt.figure(figsize=(8, 5))
pos = {
    "Obesity": (0, 1),
    "High_Blood_Sugar": (0, -1),
    "Diabetes": (2, 0)
}
nx.draw(
    graph,
    pos,
    with_labels=True,
    node_size=3500,
    font_size=11,
    arrows=True
)

plt.title("Bayesian Network for Diabetes Prediction")
plt.axis("off")
plt.show()

cpt = np.array([
    [0.98, 0.80, 0.70, 0.20],
    [0.02, 0.20, 0.30, 0.80]
])

cpt_df = pd.DataFrame(
    cpt,
    index=["No Diabetes", "Diabetes"],
    columns=[
        "No Obesity / No Sugar",
        "No Obesity / High Sugar",
        "Obesity / No Sugar",
        "Obesity / High Sugar"
    ]
)

plt.figure(figsize=(10, 5))
sns.heatmap(cpt_df, annot=True, fmt=".2f", cmap="YlOrRd")
plt.title("Diabetes Conditional Probability Table")
plt.xlabel("Condition")
plt.ylabel("Diabetes")
plt.tight_layout()
plt.show()
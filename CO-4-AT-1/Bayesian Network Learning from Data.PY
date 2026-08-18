print("SHAIK RAFIQHUDDIN - 192525129")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from pgmpy.causal_discovery import HillClimbSearch
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination

df = pd.read_csv("insurance_data.csv")

print("\nDataset:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

for column in df.columns:
    df[column] = df[column].astype(str)

print("\nLearning Bayesian Network Structure...")

hc = HillClimbSearch(
    scoring_method="bic-d",
    show_progress=False
)

learned_graph = hc.fit(df)

print("\nLearned Network Structure:")

for edge in learned_graph.causal_graph_.edges():
    print(edge)

model = DiscreteBayesianNetwork(
    learned_graph.causal_graph_.edges()
)

for column in df.columns:
    if column not in model.nodes():
        model.add_node(column)

model.fit(df)

print("\nBayesian Network Valid:")
print(model.check_model())

print("\nLearned Conditional Probability Tables:")

for cpd in model.get_cpds():
    print(cpd)

inference = VariableElimination(model)

evidence = {
    "Age": "Senior",
    "Income": "Medium",
    "Vehicle_Type": "SUV"
}

print("\nNew Customer:")
print(evidence)

result = inference.query(
    variables=["Insurance_Claim"],
    evidence=evidence
)

print("\nInsurance Claim Prediction:")
print(result)

state_names = result.state_names["Insurance_Claim"]

if "Yes" in state_names:
    yes_index = state_names.index("Yes")
    probability = result.values[yes_index]
else:
    probability = 0

print(
    f"\nProbability of Insurance Claim: "
    f"{probability * 100:.2f}%"
)

graph = nx.DiGraph()
graph.add_edges_from(model.edges())

plt.figure(figsize=(10, 7))

pos = nx.spring_layout(
    graph,
    seed=42
)

nx.draw(
    graph,
    pos,
    with_labels=True,
    node_size=3500,
    font_size=10,
    arrows=True
)

plt.title("Learned Bayesian Network Structure")
plt.axis("off")
plt.show()

claim_counts = df["Insurance_Claim"].value_counts()

plt.figure(figsize=(7, 5))

sns.barplot(
    x=claim_counts.index,
    y=claim_counts.values
)

plt.title("Insurance Claim Distribution")
plt.xlabel("Insurance Claim")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

age_claim_table = pd.crosstab(
    df["Age"],
    df["Insurance_Claim"]
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    age_claim_table,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Age vs Insurance Claim")
plt.xlabel("Insurance Claim")
plt.ylabel("Age")
plt.tight_layout()
plt.show()
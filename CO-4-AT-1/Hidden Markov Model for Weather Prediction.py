print("SHAIK RAFIQHUDDIN - 192525129")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

states = ["Sunny", "Cloudy", "Rainy"]
observations = ["Dry", "Humid", "Wet"]

state_index = {state: i for i, state in enumerate(states)}
observation_index = {observation: i for i, observation in enumerate(observations)}

transition_matrix = np.array([
    [0.70, 0.20, 0.10],
    [0.30, 0.40, 0.30],
    [0.10, 0.30, 0.60]
])

emission_matrix = np.array([
    [0.80, 0.15, 0.05],
    [0.20, 0.60, 0.20],
    [0.05, 0.25, 0.70]
])

initial_probability = np.array([
    0.50,
    0.30,
    0.20
])

observed_sequence = [
    "Dry",
    "Humid",
    "Wet",
    "Wet",
    "Humid"
]

print("Observed Sequence:")
print(observed_sequence)

time_steps = len(observed_sequence)
num_states = len(states)

viterbi = np.zeros((time_steps, num_states))
backpointer = np.zeros((time_steps, num_states), dtype=int)

first_observation = observation_index[observed_sequence[0]]

viterbi[0] = (
    initial_probability *
    emission_matrix[:, first_observation]
)

for t in range(1, time_steps):
    current_observation = observation_index[observed_sequence[t]]

    for current_state in range(num_states):
        probabilities = (
            viterbi[t - 1] *
            transition_matrix[:, current_state]
        )

        best_previous_state = np.argmax(probabilities)

        viterbi[t, current_state] = (
            probabilities[best_previous_state] *
            emission_matrix[current_state, current_observation]
        )

        backpointer[t, current_state] = best_previous_state

last_state = np.argmax(viterbi[-1])

best_path = [last_state]

for t in range(time_steps - 1, 0, -1):
    last_state = backpointer[t, last_state]
    best_path.append(last_state)

best_path.reverse()

predicted_states = [
    states[index]
    for index in best_path
]

print("\nPredicted Hidden Weather States:")

for i in range(time_steps):
    print(
        f"Day {i + 1}: "
        f"Observation = {observed_sequence[i]}, "
        f"Predicted State = {predicted_states[i]}"
    )

viterbi_df = pd.DataFrame(
    viterbi,
    columns=states,
    index=[f"Day {i + 1}" for i in range(time_steps)]
)

print("\nViterbi Probability Matrix:")
print(viterbi_df)

plt.figure(figsize=(7, 5))
sns.heatmap(
    transition_matrix,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=states,
    yticklabels=states
)
plt.title("HMM Transition Probability Matrix")
plt.xlabel("Next State")
plt.ylabel("Current State")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))
sns.heatmap(
    emission_matrix,
    annot=True,
    fmt=".2f",
    cmap="Greens",
    xticklabels=observations,
    yticklabels=states
)
plt.title("HMM Emission Probability Matrix")
plt.xlabel("Observation")
plt.ylabel("Hidden State")
plt.tight_layout()
plt.show()

predicted_numbers = [
    state_index[state]
    for state in predicted_states
]

plt.figure(figsize=(10, 5))
plt.plot(
    range(1, time_steps + 1),
    predicted_numbers,
    marker="o"
)
plt.yticks(
    range(len(states)),
    states
)
plt.xticks(range(1, time_steps + 1))
plt.xlabel("Day")
plt.ylabel("Predicted Weather")
plt.title("Predicted Hidden Weather States")
plt.grid(True)
plt.tight_layout()
plt.show()
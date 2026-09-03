print("Q3: Naive Bayes Classification")

prior_yes = 3 / 5
prior_no = 2 / 5

p_fever1_yes = 3 / 3
p_headache0_yes = 1 / 3
p_fever1_no = 0 / 2
p_headache0_no = 1 / 2

print(f"Prior P(Yes) = {prior_yes:.4f}")
print(f"Prior P(No) = {prior_no:.4f}")

print("\nConditional probabilities:")
print(f"P(Fever=1 | Yes) = {p_fever1_yes:.4f}")
print(f"P(Headache=0 | Yes) = {p_headache0_yes:.4f}")
print(f"P(Fever=1 | No) = {p_fever1_no:.4f}")
print(f"P(Headache=0 | No) = {p_headache0_no:.4f}")

numerator_yes = prior_yes * p_fever1_yes * p_headache0_yes
numerator_no = prior_no * p_fever1_no * p_headache0_no
posterior = numerator_yes / (numerator_yes + numerator_no)

print("\nFor Fever=1 and Headache=0:")
print(f"Unnormalized Yes = {numerator_yes:.4f}")
print(f"Unnormalized No = {numerator_no:.4f}")
print(f"P(Disease=Yes | Fever=1, Headache=0) = {posterior:.4f}")
print(f"Final classification: {'Disease = Yes' if posterior >= 0.5 else 'Disease = No'}")

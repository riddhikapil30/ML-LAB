#Aim: Demonstrate the working model and principle of candidate elimination algorithm.
#Program: For a given set of training data examples stored in a .CSV file, implement and demonstrate the Candidate-Elimination algorithm to output a description of the set of all hypotheses consistent with the training examples.
import pandas as pd

# Read the CSV file
data = pd.read_csv('2.csv')
concepts = data.iloc[:, :-1].values
target = data.iloc[:, -1].values

# Initialize specific and general hypotheses
specific_h = concepts[0].copy()
general_h = [['?' for _ in range(len(specific_h))] for _ in range(len(specific_h))]

def learn(concepts, target, specific_h, general_h):
    for i, concept in enumerate(concepts):
        if target[i] == "yes":
            for x in range(len(specific_h)):
                if concept[x] != specific_h[x]:
                    specific_h[x] = '?'
                    general_h[x][x] = '?'
        elif target[i] == "no":
            for x in range(len(specific_h)):
                if concept[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = '?'
        print(f"\nAfter instance {i + 1}:\nSpecific Hypothesis: {specific_h}\nGeneral Hypothesis: {general_h}")

    # Remove overly general hypotheses in G
    general_h = [g for g in general_h if g != ['?' for _ in range(len(specific_h))]]
    return specific_h, general_h

# Run the learning algorithm
s_final, g_final = learn(concepts, target, specific_h, general_h)

print("\nThe Final Specific Hypothesis:", s_final)
print("\nThe Final General Hypotheses:", g_final)



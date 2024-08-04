#Aim: Illustrate and Demonstrate the working model and principle of Find-S algorithm.
#Program: For a given set of training data examples stored in a .CSV file, implement and demonstrate the Find-S algorithm to output a description of the set of all hypotheses consistent with the training examples.

import csv
def load_data(file_name):
    data = []
    with open(file_name, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
    return data

def find_s(data):
    hypothesis = ['0'] * (len(data[0]) - 1)
    for example in data:
        if example[-1] == 'yes':
            for i in range(len(example) - 1):
                if hypothesis[i] == '0':
                    hypothesis[i] = example[i]
                elif hypothesis[i] != example[i]:
                    hypothesis[i] = '?'
    return hypothesis

def main():
    file_name = '1.csv'
    data = load_data(file_name)
    hypothesis = find_s(data)
    print("The hypothesis is:", hypothesis)

if __name__ == "__main__":
    main()
    
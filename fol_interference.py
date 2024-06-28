# fol_inference.py
import csv

import pandas as pd
from nltk.inference import ResolutionProver
from nltk.sem import Expression

read_expr = Expression.fromstring

kb = []


def load_kb_from_file(file_path='knowledge.csv'):
    data = pd.read_csv(file_path, header=None)

    for row in data[0]:
        expr = read_expr(row)
        kb.append(expr)

    # with open(file_path, 'r') as file:
    #     for line in file:
    #         # Assuming each line is a logical statement enclosed in double quotes
    #         statement = line.strip().strip('"')
    #         expr = read_expr(statement)
    #         kb.append(expr)


def add_to_kb(statement):
    # Split the statement into subject and predicate based on ' IS '
    parts = statement.split(' IS ')
    if len(parts) != 2:
        print("Error: Statement does not follow the 'subject IS predicate' format.")
        return
    subject, predicate = parts[0], parts[1]

    # Format the logical statement properly
    logical_statement = f'{predicate} ({subject})'

    # Convert the logical statement to an expression
    expr = read_expr(logical_statement)

    # Check if the expression is not already in the knowledge base
    if expr not in kb:
        # Add the expression to the knowledge base
        kb.append(expr)

        # Specify the file path to your knowledge base CSV file
        file_path = 'knowledge.csv'  # Update this to the actual path if necessary

        # Open the CSV file and append the new knowledge
        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file, quoting=csv.QUOTE_MINIMAL)
            # Write the logical statement without the outer double quotes
            writer.writerow([logical_statement])

        print(f"OK, I will remember that {subject} is {predicate}.")
    else:
        print(f"I already know that {logical_statement}.")


def check_statement(statement):
    load_kb_from_file()

    subject = statement[1].split(' IS ')[0]
    predicate = statement[1].split(' IS ')[1]

    expr = read_expr(predicate + '(' + subject + ')')

    # Check if the expression or its negation can be proved using the knowledge base
    correctAnswer = ResolutionProver().prove(expr, kb, verbose=False)
    wrongAnswer = ResolutionProver().prove(-expr, kb, verbose=False)  # Using ~ for negation

    if correctAnswer:
        return "Correct"
    elif wrongAnswer:
        return "Incorrect"
    else:
        return "I don't know"

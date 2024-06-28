import os
import aiml
import requests
import football_api
from text_similarity import setup_similarity, find_closest_match
import nltk
import csv

from nltk.sem import Expression
from nltk.inference import ResolutionProver
from nltk import load_parser

read_expr = Expression.fromstring

# Load initial knowledge base
def load_kb(path='knowledge.txt'):
    kb = []
    with open(path, 'r') as file:
        for line in file:
            if line.strip():
                kb.append(read_expr(line.strip()))
    return kb

# Check for contradictions in the KB
def check_kb(kb):
    for fact in kb:
        if not ResolutionProver().prove(fact, kb):
            return False, fact
    return True, None

# Update the KB with a new fact
def update_kb(kb, new_fact):
    # Parse the new fact to ensure it follows the correct format
    if new_fact.startswith("I know that"):
        fact = new_fact.replace("I know that", "").strip()
        player_name, role = fact.split(" is a ")
        expr = f'Player("{player_name}") -> {role}("{player_name}")'
    else:
        expr = read_expr(new_fact)
        
    if not ResolutionProver().prove(expr, kb):
        kb.append(expr)
        return True, "OK, I will remember that."
    return False, "This contradicts what I know."



# Query the KB
def query_kb(kb, query):
    expr = read_expr(query)
    if ResolutionProver().prove(expr, kb):
        return "Correct"
    elif ResolutionProver().prove(~expr, kb):
        return "Incorrect"
    else:
        return "I don't know"

# Main loop
def main():
    # Load AIML kernel
    kernel = aiml.Kernel()
    kernel.verbose(False)
    if not os.path.isfile("bot_brain.brn"):
        print("No brain file found. Learning from AIML files.")
        kernel.bootstrap(learnFiles="/Users/macbook/Desktop/RAHEEM/modules/Artificial intelligence/chatbot/venv/football.aiml", commands="load aiml b")
        kernel.saveBrain("bot_brain.brn")
    else:
        print("Loading from brain file.")
        kernel.bootstrap(brainFile="bot_brain.brn")

    # Initialize text similarity components
    tfidf_matrix, qa_pairs = setup_similarity()

    # Load logical knowledge base
    kb = load_kb()

    # Check KB for contradictions
    kb_ok, problem = check_kb(kb)
    if not kb_ok:
        print(f"Warning: Found contradiction with {problem}")
        return

    print("Welcome to the Football Chatbot! Type 'exit' to quit.")
    asking_for_live_score = False
    while True:
        input_text = input("User: ").strip()
        if input_text.lower() == "exit":
            break

        if "live score" in input_text.lower() or "live match" in input_text.lower():
            print("Bot: For live scores, please specify the team you are interested in.")
            asking_for_live_score = True
        elif asking_for_live_score:
            # If the flag is true, process the input for live scores
            print("Bot:", football_api.get_live_scores(input_text))
            asking_for_live_score = False 
        else:
            # Handle other interactions
            if input_text.startswith("I know that"):
                fact = input_text.replace("I know that", "").strip()
                success, response = update_kb(kb, fact)
                print("Bot:", response)
            elif input_text.startswith("Check that"):
                query = input_text.replace("Check that", "").strip()
                response = query_kb(kb, query)
                print("Bot:", response)
            else:
                # Default handling using AIML kernel or text similarity
                response = kernel.respond(input_text)
                if response and "Invalid input format" not in response:
                    print("Bot:", response)
                else:
                    closest_match = find_closest_match(input_text, tfidf_matrix, qa_pairs)
                    print("Bot:", closest_match or "Sorry, I can't find a relevant answer.")

if __name__ == "__main__":
    main()

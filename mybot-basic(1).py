#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic chatbot design --- for your own modifications
"""
import contextlib
from PIL import Image
import numpy as np
import os
from tensorflow import keras
from googletrans import Translator, LANGUAGES

#######################################################
# Initialise Wikipedia agent
#######################################################
import wikipedia

import fol_interference
import text_similarity

#######################################################
# Initialise weather agent
#######################################################
import json, requests

from team_names import clubs

#insert your personal OpenWeathermap API key here if you have one, and want to use this feature
APIkey = "5403a1e0442ce1dd18cb1bf7c40e776f"

#######################################################
#  Initialise AIML agent
#######################################################
import aiml
# Create a Kernel object. No string encoding (all I/O is unicode)
kern = aiml.Kernel()
kern.setTextEncoding(None)
# Use the Kernel's bootstrap() method to initialize the Kernel. The
# optional learnFiles argument is a file (or list of files) to load.
# The optional commands argument is a command (or list of commands)
# to run after the files are loaded.
# The optional brainFile argument specifies a brain file to load.
kern.bootstrap(learnFiles="mybot-basic.xml")

tfidf_matrix, qa_pairs = text_similarity.setup_similarity()


# image recogniser
def preprocessImage(filePath):
    img = Image.open(filePath).convert('RGB')  # Convert image to RGB, dropping the alpha channel if present
    img = img.resize((28, 28))
    imgArr = np.array(img) / 255.0
    imgArr = np.expand_dims(imgArr, axis=0)
    return imgArr


def classifyImage(filePath, model, classNames):
    img = preprocessImage(filePath)

    # Silence the model output
    with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
        prediction = model.predict(img)

    predictionIndex = np.argmax(prediction)
    predictionClass = classNames[predictionIndex]
    return predictionClass


def recogniseImage(filePath):
    absolutePath = os.path.abspath(filePath)
    modelPath = "./model/trained_model.h5"
    model = keras.models.load_model(modelPath)

    classification = classifyImage(absolutePath, model, clubs)
    return classification

#######################################################
# Welcome user
#######################################################
print("Welcome to this chat bot. Please feel free to ask questions from me!")
#######################################################
# Main loop
#######################################################
translator = Translator()

while True:
    # Get user input
    try:
        userInput = input("> ")
    except (KeyboardInterrupt, EOFError) as e:
        print("Bye!")
        break

    # Pre-process user input and determine response agent (if needed)
    responseAgent = 'aiml'

    # Activate selected response agent
    if responseAgent == 'aiml':
        answer = kern.respond(userInput)

    # Post-process the answer for commands
    if answer[0] == '#':
        params = answer[1:].split('$')
        cmd = int(params[0])

        if cmd == 0:
            print(params[1])
            break

        elif cmd == 1:
            try:
                wSummary = wikipedia.summary(params[1], sentences=3, auto_suggest=False)
                print(wSummary)
            except:
                print("Sorry, I do not know that. Be more specific!")

        elif cmd == 2:
            succeeded = False
            api_url = "http://api.openweathermap.org/data/2.5/weather?q="
            response = requests.get(api_url + params[1] + "&units=metric&APPID=" + APIkey)
            if response.status_code == 200:
                response_json = json.loads(response.content)
                if response_json:
                    t = response_json['main']['temp']
                    tmi = response_json['main']['temp_min']
                    tma = response_json['main']['temp_max']
                    hum = response_json['main']['humidity']
                    wsp = response_json['wind']['speed']
                    wdir = response_json['wind']['deg']
                    conditions = response_json['weather'][0]['description']
                    print(f"The temperature is {t} °C, varying between {tmi} and {tma} at the moment, humidity is {hum}%, wind speed {wsp} m/s, {conditions}")
                    succeeded = True
            if not succeeded:
                print("Sorry, I could not resolve the location you gave me.")

        elif cmd == 99:
            closest_match = text_similarity.find_closest_match(userInput, tfidf_matrix, qa_pairs)
            response = closest_match or "Sorry, I can't find a relevant answer."

            # ask the user if they want text to speech
            print("Do you want to enable speech on this question? (yes/no): ")
            speech_on = input("> ").strip().lower()

            if speech_on == "yes":
                print(response)
                os.system(f"say {response}")

            # Ask the user if they want the response translated to Spanish
            print("Do you want the response translated to Spanish? (yes/no): ")
            translate_input = input("> ").strip().lower()

            if translate_input == "yes":
                # Translate the response to Spanish
                translated_response = translator.translate(response, dest='es').text
                print(translated_response)
                os.system(f"say {translated_response}")

            else:
                print(response)

        elif cmd == 22:
            fol_interference.add_to_kb(params[1])

        elif cmd == 23:
            print(fol_interference.check_statement(params))

        elif cmd == 100:
            print("Enter file path for image: ")
            filePathInput = input("> ")
            print(recogniseImage(filePathInput))

    else:
        print(answer)
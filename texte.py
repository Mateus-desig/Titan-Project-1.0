import pyttsx3
engine = pyttsx3.init()

voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)


engine.say("Eu vou falar esse texto.")
engine.runAndWait()
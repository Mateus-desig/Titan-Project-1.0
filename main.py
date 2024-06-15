# Our main file.

from vosk import Model, KaldiRecognizer
import os
import pyaudio
import pyttsx3
import json
import core
from nlu.classfier import classify

# Síntese de fala.
engine = pyttsx3.init()

voices = engine.getProperty('voices')
engine.setProperty('voice', voices[-1].id)


def speak(text):
    engine.say(text)
    engine.runAndWait()

def evaluate(text):
    # Reconhecer entidade do texto.
    entity = classify(text)

    if entity == 'time|getTime':
        speak(core.SystemInfo.get_time())
    elif entity == 'time|getDate':
        speak(core.SystemInfo.get_date())

    # Abrir programas.
    elif entity == 'open|notepad':
        speak('Abrindo o bloco de notas.')
        os.system('notepad.exe')

    elif entity == 'open|chrome':
        speak('Abrindo o chrome.')
        os.system('"C:/Program Files/Google/Chrome/Application/chrome.exe"')

    elif entity == 'open|minecraft':
        speak('Abrindo o minecraft.')
        os.system('"C:/Users/utrac/AppData/Roaming/Microsoft/Windows/Start Menu/Programs/Overwolf.exe"')


    print('text: {} Entity: {}'.format(text, entity))



# Reconhecimento de fala.
model = Model('model')
rec = KaldiRecognizer(model, 16000)

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
stream.start_stream()

# Loop do reconhecimento de fala.
while True:
    data = stream.read(8192)
    if len(data) == 0:
        break
    if rec.AcceptWaveform(data):
        result = rec.Result()
        result = json.loads(result)

        if result is not None:
            text = result['text']
            evaluate(text)

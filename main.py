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

            # Reconhecer entidade do texto.
            entity = classify(text)

            # speak(text)

            if entity == 'time\getTime':
                speak(core.SystemInfo.get_time())

            print(text)
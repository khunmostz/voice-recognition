from pydub import AudioSegment

audio = AudioSegment.from_mp3("./output.mp3")

audio.export("output.wav", format="wav")

import speech_recognition as sr

r = sr.Recognizer()

with sr.AudioFile('output.wav') as source:
    audio = r.record(source)

try:
    text = r.recognize_google(audio, language="th-TH")
    print("Sphinx thinks you said: " + text)
except sr.UnknownValueError:
    print("Sphinx could not understand audio")
except sr.RequestError as e:
    print("Sphinx error; {0}".format(e))
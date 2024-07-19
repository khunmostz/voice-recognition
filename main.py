from google_speech import Speech

text = "สินเชื่อมีกี่มีแบบหรอครับ"
lang = "th"
speech = Speech(text, lang)
speech.play()

# you can also apply audio effects while playing (using SoX)
# see http://sox.sourceforge.net/sox.html#EFFECTS for full effect documentation
# sox_effects = ("speed", "1.5")
# speech.play(sox_effects)

speech.save("./mp3/output.mp3")

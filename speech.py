
import speech_recognition as speech_recog
# Creating a recording object to store input
rec = speech_recog.Recognizer()
# Importing the microphone class to check availabiity of microphones
mic_test = speech_recog.Microphone()
# List the available microphones
speech_recog.Microphone.list_microphone_names()
# We will now directly use the microphone module to capture voice input. Specifying the second microphone to be used for a duration of 3 seconds. The algorithm will also adjust given input and clear it of any ambient noise
with speech_recog.Microphone(device_index=1) as source:
    rec.adjust_for_ambient_noise(source, duration=3)
    print("Reach the Microphone and say something!")
    audio = rec.listen(source)

# Use the recognize function to transcribe spoken words to text
try:
    print("I think you said: \n" + rec.recognize_google(audio))
except Exception as e:
    print(e)

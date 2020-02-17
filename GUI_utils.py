from gtts import gTTS
from playsound import playsound
from googletrans import Translator
import os
import speech_recognition as sr
import csv

trans = Translator()

## Util Programs:
def init():
    data = {}
    with open('data.csv') as f:
        read = csv.reader(f)
        count=0
        for i in read:
            data[count]=(i[0],i[1],i[2],i[3])
            count+=1
    return data

def audio2text(lang):

    try:
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print(' Speak Anything : ',end='')
            audio = r.listen(source)
            text= r.recognize_google(audio, language=lang[0])
            t = trans.translate(text, dest='en')
            result = t.text
            print(text, t.text)                 #change
    except:
        result = ''
        text = ''
    return result, text


def text2native(text, lang):

    text = trans.translate(text, dest=lang[1])
    print(text.text)
    text1=gTTS(text=text.text,lang=lang[2],slow=False)
    text1.save('Voice1.mp3')
    playsound('Voice1.mp3')
    os.remove('Voice1.mp3')
    return text.text





'''import time

t = time.time()
language = init()
text = audio2text(language['English (India)'])
text2native(text, language['Hindi (India)'])
print(time.time()-t)'''

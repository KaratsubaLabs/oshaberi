import requests
from config import ONIGIRI_URL, OPENTTS_URL

def define_en_get(word):
    resp = requests.get(f"{ONIGIRI_URL}/define/en/{word}")
    if resp:
        print(resp.json())


# TODO look into pyaudio or sm to play the response audio

TTS_VOICE = "coqui-tts:en_vctk"
def synthesis_get(text):
    # TODO text should be sanitised
    resp = requests.get(f"{OPENTTS_URL}/tts?voice={TTS_VOICE}&text={text}")
    print(resp)

if __name__ == "__main__":
    # define_en_get("hello")
    synthesis_get("hello world")

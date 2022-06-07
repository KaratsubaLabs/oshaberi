import requests
from config import API_URL

def define_en_get(word):
    resp = requests.get(f"{API_URL}/define/en/{word}")
    if resp:
        print(resp.json())


if __name__ == "__main__":
    define_en_get("hello")


import nltk

DOWNLOAD_DIR="./venv/nltk_data"

resources = ["stopwords", "punkt"]
for resource in resources:
    nltk.download(resource, download_dir=DOWNLOAD_DIR)

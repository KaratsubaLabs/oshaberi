
import nltk

DOWNLOAD_DIR="./venv/nltk_data"

resources = ["stopwords", "punkt", "averaged_perceptron_tagger", "maxent_ne_chunker", "words"]
for resource in resources:
    nltk.download(resource, download_dir=DOWNLOAD_DIR)


# TODO this is prob not needed lol
from pipeop import pipes

import preprocess

test_string = "very simple python chatbot to suck less at nlp"

@pipes
def run():
    print(test_string
        >> preprocess.tokenize
        >> preprocess.filter_stopwords
        >> preprocess.stem
    )

run()


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# TODO this is prob not needed lol
from pipeop import pipes
import preprocess
from dataset import IntentDataset

test_data = [
    ("advice", "In my younger and more vulnerable years my father gave me some advice that I've been turning over in my mind ever since."),
    ("critism", "Whenever you feel like criticizing any one, he told me, just remember that all the people in this world haven't had the advantages that you've had."),
    ("communication", "He didn't say any more but we've always been unusually communicative in a reserved way, and I understood that he meant a great deal more than that.")
]

@pipes
def run():
    word_dict = []
    xy = []

    # load data
    for (tag, test_string) in test_data:
        tokenized = test_string >> preprocess.tokenize
        xy.append((tokenized, tag))

        word_dict.extend(
            tokenized >> preprocess.filter_stopwords >> preprocess.stem
        )

    word_dict = word_dict >> set >> sorted
    print(word_dict)
    print(xy)

    # build training data
    x_data = np.array([
        preprocess.bag_words(tokenized, word_dict) for (tokenized, tag) in xy
    ])
    y_data = np.array([tag for (tokenized, tag) in xy])
    dataset = IntentDataset(x_data, y_data)

    batch_size = 8
    num_workers = 2
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

run()

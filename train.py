
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pipeop import pipes

import preprocess
from dataset import IntentDataset
from nn import NeuralNet
from reader import import_intents, get_tags
import config


@pipes
def train(batch_size=8, num_workers=2, hidden_size=8, learning_rate=0.001, training_epochs=1000):
    # load data
    intents = import_intents(config.INTENTS_FILEPATH)
    tags = get_tags(intents)

    word_dict = []
    xy = []

    # process intents for training
    for intent in intents:
        tag = intent["tag"]
        for utterance in intent["utterances"]:
            tokenized = (
                utterance
                >> preprocess.tokenize
                >> preprocess.filter_stopwords
                >> preprocess.stem
            )
            xy.append((tokenized, tag))
            word_dict.extend(tokenized)

    word_dict = word_dict >> set >> sorted
    print(word_dict)
    print(xy)
    print(tags)

    # build training data
    x_data = np.array([
        preprocess.bag_words(tokenized, word_dict) for (tokenized, tag) in xy
    ])
    y_data = np.array([tags.index(tag) for (tokenized, tag) in xy])
    print(x_data)
    print(y_data)
    dataset = IntentDataset(x_data, y_data)

    # build dataloader
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    # build neural net
    input_size = len(word_dict)

    output_size = len(tags)
    model = NeuralNet(input_size, hidden_size, output_size).to(config.MODEL_DEVICE)

    # start training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(training_epochs):
        for (words, labels) in loader:
            words = words.to(config.MODEL_DEVICE)
            labels = labels.to(config.MODEL_DEVICE)

            # forward pass
            outputs = model(words)
            loss = criterion(outputs, labels)

            # backwards pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 100 == 0:
            print(f"epoch={epoch}/{training_epochs} loss={loss.item():.4f}")

    print(f"final loss={loss.item():.4f}")

    # save training data to file
    model_data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "word_dict": word_dict,
        "tags": tags
    }
    torch.save(model_data, config.MODEL_FILEPATH)
    print("successfully saved model data to file")


if __name__ == "__main__":
    train()

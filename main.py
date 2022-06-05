
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# TODO this is prob not needed lol
from pipeop import pipes
import preprocess
from dataset import IntentDataset
from nn import NeuralNet

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
    # TODO make this reference index of tag
    y_data = np.array([i for i in range(len(test_data))])
    dataset = IntentDataset(x_data, y_data)

    # build dataloader
    batch_size = 8
    num_workers = 2
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    # build neural net
    input_size = len(word_dict)
    hidden_size = 8
    output_size = 3
    device = 'cpu'
    model = NeuralNet(input_size, hidden_size, output_size).to(device)

    # start training
    learning_rate = 0.001
    training_epochs = 1000
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(training_epochs):
        for (words, labels) in loader:
            words = words.to(device)
            labels = labels.to(device)

            # forward pass
            outputs = model(words)
            loss = criterion(outputs, labels)

            # backwards pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 100 == 0:
            print(f'epoch={epoch}/{training_epochs} loss={loss.item():.4f}')

    print(f'final loss={loss.item():.4f}')


run()

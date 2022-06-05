import sys
import torch

from nn import NeuralNet
import preprocess

if len(sys.argv) != 2:
    raise ValueError("pass in an utterance")

query = sys.argv[1]

# load model data and rebuild neural net
model_filepath = "./out/intents.pth"
model_data = torch.load(model_filepath)

model_state = model_data["model_state"]
input_size = model_data["input_size"]
hidden_size = model_data["hidden_size"]
output_size = model_data["output_size"]
word_dict = model_data["word_dict"]
tags = model_data["tags"]
device = "cpu"

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# preprocess query
x = preprocess.tokenize(query)
x = preprocess.stem(x)
x = preprocess.bag_words(x, word_dict)
x = x.reshape(1, x.shape[0])
x = torch.from_numpy(x)

output = model(x)
_, predicted = torch.max(output, dim=1)
tag = tags[predicted.item()]

print(tag)


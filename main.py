import sys
import torch
from pipeop import pipes

from nn import NeuralNet
import preprocess
import config

if len(sys.argv) != 2:
    raise ValueError("pass in an utterance")

query = sys.argv[1]

# load model data and rebuild neural net
model_data = torch.load(config.MODEL_FILEPATH)

model_state = model_data["model_state"]
input_size = model_data["input_size"]
hidden_size = model_data["hidden_size"]
output_size = model_data["output_size"]
word_dict = model_data["word_dict"]
tags = model_data["tags"]

model = NeuralNet(input_size, hidden_size, output_size).to(config.MODEL_DEVICE)
model.load_state_dict(model_state)
model.eval()

@pipes
def preprocess_query(query):
    x = (query
        >> preprocess.tokenize
        >> preprocess.stem
        >> preprocess.bag_words(word_dict)
    )
    x = x.reshape(1, x.shape[0])
    return torch.from_numpy(x)


preprocessed = preprocess_query(query)
# TODO catch a tensor that is all zero
print(preprocessed)
output = model(preprocessed)
_, predicted = torch.max(output, dim=1)
tag = tags[predicted.item()]

probs = torch.softmax(output, dim=1)
prob = probs[0][predicted.item()]

if prob.item() > config.CONFIDENCE_THRESHOLD:
    print(f"[prob={prob.item():.4f}] {tag}")
else:
    print("query not understood")



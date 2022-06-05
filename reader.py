import json
from pipeop import pipes


def import_intents(filepath):
    with open(filepath, "r") as filehandle:
        data = json.load(filehandle)

    # TODO probably do some validation on the input
    return data["intents"]


@pipes
def get_tags(intents):
    return (
        [intent["tag"] for intent in intents]
        >> set
        >> sorted
    )


if __name__ == "__main__":
    import_intents("./data/intents.json")

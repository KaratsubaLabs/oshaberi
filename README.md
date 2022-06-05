
# oshaberi

very simple python chatbot to suck less at nlp

## RUNNING FOR DEVELOPMENT

First create venv and install dependencies
```
$ virtualenv --python=<path to python3.7> venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

Next, download required nltk data
```
$ python bin/download.py
```

## TODO

- [ ] properly load intents data from json
- [ ] add type annotations
- [ ] do some benchmarking (maybe)
- [ ] implement simple user interaction
- [ ] python linter

## RESOURCES

using these resources
- [nlp basics](https://realpython.com/nltk-nlp-python/)
- [playlist to create chatbot](https://www.youtube.com/watch?v=RpWeNzfSUHw&list=PLqnslRFeH2UrFW4AUgn-eY37qOAWQpJyg)

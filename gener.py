import numpy as np
from keras.models import model_from_yaml
from random import randint
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

with open("sonnets.txt") as corpus_file:
    corpus = corpus_file.read()
print("Loaded a corpus of {0} characters".format(len(corpus)))

# Get a unique identifier for each char in the corpus, then make some dicts to ease encoding and decoding
chars = sorted(list(set(corpus)))
encoding = {c: i for i, c in enumerate(chars)}
decoding = {i: c for i, c in enumerate(chars)}

# Some variables we'll need later
num_chars = len(chars)
sentence_length = 50
corpus_length = len(corpus)

with open("model.yaml") as model_file:
    architecture = model_file.read()

model = model_from_yaml(architecture)
model.load_weights("weights.hdf5")
model.compile(loss='categorical_crossentropy', optimizer='adam')


seed = randint(0, corpus_length - sentence_length)
seed_phrase = corpus[seed:seed + sentence_length]

X = np.zeros((1, sentence_length, num_chars), dtype=np.bool)
for i, character in enumerate(seed_phrase):
    X[0, i, encoding[character]] = 1


generated_text = ""
for i in range(500):
    prediction = np.argmax(model.predict(X, verbose=0))

    generated_text += decoding[prediction]

    activations = np.zeros((1, 1, num_chars), dtype=np.bool)
    activations[0, 0, prediction] = 1
    X = np.concatenate((X[:, 1:, :], activations), axis=1)

print(generated_text)

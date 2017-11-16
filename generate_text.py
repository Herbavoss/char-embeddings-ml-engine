from keras.models import load_model
from tensorflow.python.lib.io import file_io

import numpy as np

maxlen = 40

input_data_file = file_io.FileIO('data/input.txt', mode='r')
text = input_data_file.read()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

model = load_model('output/gcloud_trained.hdf5')
print model.summary()

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-6) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

generated = ''
diversity = 0.4

# sentence must be maxlen long
#sentence = '0.03% of the world online gambling marke'
sentence = '$12.5M USD denominated in ETH in our tok'
#sentence = 'Abstract. A purely peer-to-peer version '

generated += sentence
print('----- Generating with seed: "' + sentence + '"')

for diversity in [0.2, 0.5]:
    print()
    print('----- diversity:', diversity)
    for i in range(2000):
        x = np.zeros((1, maxlen), dtype=np.int)
        for t, char in enumerate(sentence):
            x[0, t] = char_indices[char]

        preds = model.predict(x, verbose=0)[0][0]
        next_index = sample(preds, diversity)
        next_char = indices_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

    print generated
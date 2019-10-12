import random
import numpy as np
from keras.engine.saving import model_from_json
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM, Dropout
from keras.optimizers import RMSprop
import librosa
from pysndfx import AudioEffectsChain
import logging

logging.getLogger('tensorflow').disabled = True

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")


total_notes = ""
for i in range(1, 10):
    with open("music/song" + str(i) + ".txt", "r") as file:
        text = file.read()

    notes = [float(num) for num in text.split(" ")]
    total_notes += text + " 0 0 0 0 0 "

all_notes = [int(num) for num in total_notes[:-1].split(' ')]
print(len(all_notes), "notas.")

num_notes = 25
maxlen = 20
step = 1
verses = []
next_note = []
for i in range(0, len(all_notes) - maxlen, step):
    verses.append(all_notes[i: i + maxlen])
    next_note.append(all_notes[i + maxlen])

x = np.zeros((len(verses), maxlen, num_notes), dtype=np.bool)
y = np.zeros((len(verses), num_notes), dtype=np.bool)

for i, verse in enumerate(verses):
    for j, note in enumerate(verse):
        x[i, j, note] = 1
    y[i, next_note[i]] = 1


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate_music(length, diversity):
    # Get random starting text
    start_index = random.randint(0, len(all_notes) - maxlen - 1)
    generated = []
    sentence = all_notes[start_index: start_index + maxlen]
    generated += sentence
    for i in range(length):
        x_pred = np.zeros((1, maxlen, num_notes), dtype=np.bool)
        for t, note in enumerate(sentence):
            x_pred[0, t, note] = 1.

        preds = model.predict(x_pred, verbose=0)[0]
        next_note = sample(preds, diversity)

        generated += [next_note]
        sentence = sentence[1:] + [next_note]
    return generated


# model = Sequential()
# model.add(LSTM(256, input_shape=(maxlen, num_notes)))
# model.add(Dropout(0.2))
# model.add(Dense(num_notes))
# model.add(Activation('softmax'))
# optimizer = RMSprop(lr=0.01)
# model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#
# model.fit(x, y, batch_size=64, epochs=10)

music_gen = generate_music(150, 0.5)


# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("model.h5")
# print("Saved model to disk")


notes_freq = {
    0: 0,
    1: 130.81,
    2: 138.59,
    3: 146.83,
    4: 155.56,
    5: 164.81,
    6: 174.61,
    7: 185,
    8: 196,
    9: 207.65,
    10: 220,
    11: 233.08,
    12: 246.94,
    13: 261.63,
    14: 277.18,
    15: 293.67,
    16: 311.13,
    17: 329.63,
    18: 349.23,
    19: 369.99,
    20: 392,
    21: 415.30,
    22: 440,
    23: 466.16,
    24: 493.88,
}
sr = 22050
T = 0.4 * len(music_gen)
t = np.linspace(0, T, int(T * sr), endpoint=False)

freqs = []
notes_per_second = int(T * sr / len(music_gen))
for i in music_gen:
    freq = notes_freq[i]
    freqs.append([freq] * notes_per_second)

freqs = np.array(freqs).reshape(1, -1)
diff = int(T * sr) - freqs.shape[1]

X = 0.5 * np.sin(2 * np.pi * freqs[0] * t[:freqs.shape[1]])

fx = (
    AudioEffectsChain()
    .reverb(reverberance=40, room_scale=30)
    .lowpass(700)
    .tremolo(20)
    .normalize()
)

X = fx(X)

librosa.output.write_wav('song_generated.wav', X, sr)
print("Audio saved!")
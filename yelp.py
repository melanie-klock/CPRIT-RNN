# A RNN MODEL FOR ASSESSING QUALITY OF PATIENT FALL REPORTS
# Authors: Melanie Klock
# CPRIT Summer Fellowship Program at the UTHealth Science Center in Houston, TX
# Biomedical Informatics, Dr. Gong's Lab

# TO-DO:
#   1. improve accuracy
#   2. try with larger data sample

from keras.layers import Embedding, Dense, GRU
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
# from nltk.corpus import stopwords

import numpy as np
import pandas as pd

# get data
df = pd.read_csv('practice_data.csv', encoding="ISO-8859-1")
df = df.values
dataset = df[:, 0]
labels = df[:, 1:14]

# tokenize
tokenizer = Tokenizer(num_words=200)
tokenizer.fit_on_texts(dataset)
sequences = tokenizer.texts_to_sequences(dataset)

# variables:
embedding_dim = 128
input_vocab = len(tokenizer.word_index) + 1
maxLength = 100

# make sequences same length
padded_sequences = np.array(pad_sequences(sequences, maxlen=maxLength))

# build model
model = Sequential()
model.add(Embedding(input_vocab, embedding_dim, input_length=maxLength))
model.add(GRU(embedding_dim, dropout=0.9, return_sequences=True))
model.add(GRU(embedding_dim, dropout=0.9))
model.add(Dense(13, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  # bc only 2 classes

# train model
history = model.fit(padded_sequences, np.array(labels), validation_split=0.3, epochs=3)

scores = model.evaluate(padded_sequences, np.array(labels))
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# visualize performance
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and Validation accuracy compared')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation loss compared')
plt.legend()

# plt.show()

# calculate predictions
# predictions = np.round(model.predict(padded_sequences))
# for p in predictions:
#    print("Categories: ", p)
#    print("Score: ", np.round(sum(p)/len(p), 2))


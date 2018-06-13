from keras.layers import Embedding, LSTM, Dense, Conv1D, MaxPooling1D, Dropout, Activation
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd

# get columns
df = pd.read_csv('MAUDE_2008_2016_review.csv')
iden = df['ID']
data = df['REPORT']
labels = df['HIT']

# fix reproducibility
np.random.seed(7)

# pre-processing
tokenizer = Tokenizer(num_words=1000)  # ignore all except 5 most common words
tokenizer.fit_on_texts(data)
sequences = tokenizer.texts_to_sequences(data)

padded_sequences = pad_sequences(sequences, maxlen=300)

# build model
model = Sequential()
model.add(Embedding(1000, 128, input_length=300))  # embedding layer with vocab size, vector size, & length of sequences
model.add(Dropout(0.2))
model.add(Conv1D(64, 5, activation='relu'))
model.add(MaxPooling1D(pool_size=4))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))  # LSTM layer with dropout to increase robustness of NN
model.add(Dense(1, activation='sigmoid'))  # Dense layer to have output of size 1
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# train model
model.fit(padded_sequences, np.array(labels), validation_split=0.2, epochs=3)

scores = model.evaluate(padded_sequences, np.array(labels))
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# calculate predictions
predictions = model.predict(padded_sequences)
# round predictions
# rounded = [round(x[0]) for x in predictions]
# print(rounded)
print(predictions)

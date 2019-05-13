#!/usr/bin/env python
# coding: utf-8

# Tensorflow
import tensorflow as tf
from tensorflow.python.keras.callbacks import TensorBoard
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)

# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Other
from utils import read_data
from time import time
from models import conv_lstm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = read_data('data_cleaned.csv', first=False)

print(df.head(10))

tk = Tokenizer()
tk.fit_on_texts(df['text'])

sequences = tk.texts_to_sequences(df['text'])
data = pad_sequences(sequences, maxlen=100)
print(data.shape)

X_train, X_test, y_train, y_test = train_test_split(data, df['stars'], test_size=0.25, random_state=1)

# Build neural network with LSTM and CNN
vocabulary_size =  len(tk.word_counts.keys())+1
max_words = 100
embedding_size = 128
model = lstm(vocabulary_size, embedding_size, max_words)

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

# Fit and evaluate
hist = model.fit(X_train, y_train, validation_split=0.3, batch_size=1024, epochs=3, callbacks=[tensorboard])
accuracy = model.evaluate(X_test, y_test, verbose=0)
print("accuracy : ", accuracy[1])

# Save model and weight
model.save_weights('trained_model_weights.h5')
with open('architecture.json', 'w') as f:
    f.write(model.to_json())

pred = model.predict(X_test)
y_pred = []
for x in pred:
    if x[0] >= 0.6:
        y_pred.append(1)
    else:
        y_pred.append(0)

matrix = confusion_matrix(y_test, y_pred)
print(matrix)






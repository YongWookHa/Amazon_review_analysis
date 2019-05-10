from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Dropout
from keras.layers.embeddings import Embedding

# convolutional layer를 추가한 LSTM 모델
def conv_lstm(vocabulary_size, embedding_size, max_word):
    model = Sequential()
    model.add(Embedding(vocabulary_size, embedding_size, input_length=max_word))
    model.add(Dropout(0.2))
    model.add(Conv1D(64, 5, activation='relu'))
    model.add(MaxPooling1D(pool_size=3))
    model.add(LSTM(100, dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


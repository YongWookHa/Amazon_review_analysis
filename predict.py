from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

def load_model(architecture, weights):
    # pre-trained 모델 load
    with open(architecture) as f:
        model_json = f.read()
        model = model_from_json(model_json)
    model.load_weights(weights)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def sentiment_predict(model, li):
    # LSTM 입력 데이터 형식 (word token)으로 변환
    tk = Tokenizer()
    tk.fit_on_texts(li)
    sequences = tk.texts_to_sequences(li)
    data = pad_sequences(sequences, maxlen=100)

    # 모델 output 결과 계산
    Y_pred = model.predict(data)
    res = np.average(Y_pred)

    return res
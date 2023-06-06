import re
import pandas as pd
import numpy as np 
# import matplotlib.pyplot as plt
# import urllib.request
from collections import Counter
from konlpy.tag import Mecab
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from konlpy.tag import Okt
from tensorflow.keras.models import load_model

class psng_predict_all:
    def __init__(self, df):
        self.df = df
        path = ''
        self.loaded_model = load_model(f'{path}new_model_01_v2.h5')
        train_data = pd.read_csv(f'{path}train_data_main.csv', encoding='euc-kr')
        test_data = pd.read_csv(f'{path}test_data_main.csv', encoding='euc-kr')
        X_train = train_data['tokenized'].apply(eval).values
        self.X_test = test_data['tokenized'].apply(eval).values
        self.y_test = np.array(test_data['label'].values)
        self.X_test_tkd = None
        vocab_size = 87912
        self.tokenizer = Tokenizer(vocab_size, oov_token='OOV')
        self.tokenizer.fit_on_texts(X_train)
        self.max_len = 250

    def preprocessing(self):
        okt = Okt()
        R_frm = self.df.copy()
        R_frm['리뷰'] = R_frm['리뷰'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", regex=True)
        R_frm['리뷰'].replace('', np.nan, inplace=True)
        R_frm['리뷰'] = R_frm['리뷰'].astype(str)
        stopwords = ['도', '는', '다', '의', '가', '이', '은',
                     '한', '에', '하', '고', '을', '를', '인', '듯',
                     '과', '와', '네', '들', '듯', '지', '임', '게',
                     '는', '이', '했', '슴', '음', '것', '거', '로',
                     '들', '거', '곳', '분', '원', '더', '왜', '해',
                     '수', '할', '그', '함', '돈', '번', '두', '개',
                     '건', '내', '저', '만', '갈', '걸', '제', '명',
                     '분',
                     '인데', '이가', '했었', '해서',
                     '습니다', '했는데', '입니다']

        okt = Okt()
        R_frm['tokenized'] = R_frm['리뷰'].apply(okt.pos)
        R_frm['tokenized'] = R_frm['tokenized'].apply(
            lambda x: [word for word, shape in x if shape in ['Verb', 'Adjective', 'Noun', 'VerbPrefix'] if word not in stopwords])
        R_pred = R_frm['tokenized'].values
        return R_pred

    def model_test(self):
        tokenizer = self.tokenizer
        X_test = tokenizer.texts_to_sequences(self.X_test)
        y_test = self.y_test
        X_test = pad_sequences(X_test, maxlen=self.max_len)
        print("\n 테스트 정확도: %.4f" % (self.loaded_model.evaluate(X_test, y_test)[1]))

    def word_index(self):
        tokenizer = self.tokenizer
        print(tokenizer.word_index)

    def predict(self):
        R_pred = self.preprocessing()
        max_len = self.max_len
        tokenizer = self.tokenizer
        result = []
        pred = tokenizer.texts_to_sequences(R_pred)
        pred = pad_sequences(pred, maxlen=max_len)
        score = self.loaded_model.predict(pred)
        return score

    def prediction(self):
        result = []
        score = self.predict()
        for num in range(len(score)):
            if score[num][0] > 0.9:
                result.append('0')
            elif score[num][0] < 0.1:
                result.append('1')
            else:
                result.append('2')
        return result
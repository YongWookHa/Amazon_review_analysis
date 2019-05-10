# -*- coding: utf-8 -*-
from wordcloud import WordCloud
import re
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

class ngram:
    def __init__(self, n, li):
        self.n = n;  # (1: unigram, 2: bigram, 3:trigram)
        self.data = li
        self.freq_dict = dict()  # 빈도를 저장하는 dict
        self.SSword = dict()
        self.total_words = 0  # 총 단어 수
        self.build_nGram()

    def build_nGram(self):
        # print(self.n, "-gram의 학습을 시작합니다.")
        # print("시스템에 따라 시간이 오래 걸릴 수 있습니다.")
        eng = re.compile('[^ a-zA-Z0-9]+')  # 알파벳과 띄어쓰기를 제외한 모든 글자

        for line in self.data:
            line = eng.sub('', line)  # 알파벳과 띄어쓰기를 제외한 모든 부분 제거
            li = line.split()  # word list로 변환

            if li[0] in self.SSword:
                self.SSword[li[0]] += 1
            else:
                self.SSword[li[0]] = 1

            self.total_words += len(li)

            until = None if self.n == 1 else -self.n + 1
            for i, word in enumerate(li[:until]):
                if word in self.freq_dict:  # 이미 dict에 들어 있을 경우
                    if self.n == 1:  # unigram의 경우 후보군 없이 빈도 카운팅
                        self.freq_dict[word] += 1
                    else:
                        candidate = tuple(li[i + 1:i + self.n])  # word 다음에 오는 단어들
                        if candidate in self.freq_dict[word]:
                            self.freq_dict[word][candidate] += 1
                        else:
                            self.freq_dict[word][candidate] = 1  # 2 dimensional dictionary
                else:  # 새로 추가되는 단어
                    if self.n == 1:
                        self.freq_dict[word] = 1
                    else:
                        self.freq_dict[word] = dict()
                        candidate = tuple(li[i + 1:i + self.n])  # word 다음에 오는 단어들
                        self.freq_dict[word][candidate] = 1  # 2 dimensional dictionary


    def smoothing(self, w):
        M, P = 100, 0.5  # Heuristic 하게 설정 - smoothing 사용하지 않으려면 M, P 모두 0으로 세팅
        if self.n == 1:
            sorted_items = sorted(self.freq_dict.items(), key=lambda kv: -kv[1])[:100]  # 편의상 100개만 반환
            sorted_items = [(x[0], (x[1] + M * P) / (self.total_words + M)) for x in sorted_items]
        else:
            if w is None:
                raise ValueError("w cannot be None when it's not unigram")
            try:
                sorted_items = sorted(self.freq_dict[w].items(), key=lambda kv: -kv[1])[:100]
            except KeyError:
                return [(('말을', '했다'), 1)]
            sorted_items = [(x[0], (x[1] + M * P) / (self.total_words + M)) for x in sorted_items]
        return sorted_items

    def get_freq_list(self, w=None):  # 단어와 빈도를 저장한 list 반환
        freq_list = self.smoothing(w)
        return freq_list

    def get_SSword(self):
        sorted_items = sorted(self.SSword.items(), key=lambda kv: -kv[1])[:3]
        return [x[0] for x in sorted_items]

    def gen_wordcloud(self, li, color="white"):
        wc = WordCloud(  # wordcloud 설정
            font_path='./NanumGothic.ttf',
            relative_scaling=0.2,
            background_color=color,
            width=800,
            height=800
        )
        d = dict()
        exception = ["and", "is", "the", "of", "a", "to", "I", "for", "at", "with", "from", "that", "by", "in", "on", "it", "this"]
        for x in li:
            if x[0] in exception:  # 무의미하지만 빈도가 높은 단어 제외
                continue
            d[x[0]] = x[1]

        wordcloud = wc.generate_from_frequencies(d)
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.show()
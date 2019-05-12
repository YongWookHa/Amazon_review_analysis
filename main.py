from crawler import get_review
from predict import sentiment_predict, load_model
from word_cloud import ngram

if __name__ == "__main__":
    model = load_model("architecture.json", "trained_model_weights.h5")

    while True:
        url = input("Enter Amazon Product Url- (quit for q) ")
        if url == 'q':
            break
        avg_stars, long_reviews = get_review(url)  # url을 입력받아 해당 상품의 review 추출

        res = sentiment_predict(model, long_reviews)  # 학습된 LSTM 모델을 통해 분류결과 출력
        print("result : ", res)
        print("avg_stars : ", avg_stars)

        unigram = ngram(1, long_reviews)  # review에 자주 사용된 단어를 1-gram으로 추출
        words = unigram.get_freq_list()
        if res >= 0.6:
            unigram.gen_wordcloud(words, (36, 120, 255))  # blue background
        else:
            unigram.gen_wordcloud(words, (255, 18, 18))  # red background

    # example of good sentiment
    # https://www.amazon.com/AcuRite-Humidity-Thermometer-Hygrometer-Indicator/dp/B0013BKDO8/ref=br_asw_pdt-2?pf_rd_m=ATVPDKIKX0DER&pf_rd_s=&pf_rd_r=42W413EARKNAFRYM8VJG&pf_rd_t=36701&pf_rd_p=ebb28e10-c446-456a-ac5d-f251207d3750&pf_rd_i=desktop

    # bad sentiment
    # https://www.amazon.com/Home-Zone-Stainless-Rectangular-Removable/dp/B01H6CJ7HQ/ref=sr_1_56?keywords=garbage&qid=1557376063&refinements=p_72%3A2661621011&rnid=2661617011&s=gateway&sr=8-56
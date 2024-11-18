import json
import warnings
warnings.filterwarnings(action="ignore")  # 경고 제거

import pandas as pd
import re  # 정규식 모듈
from konlpy.tag import Okt  # 형태소 분석

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression  # 로지스틱 회귀모델
from sklearn.model_selection import GridSearchCV  # 하이퍼 매개변수 최적값 구하기 모듈
from sklearn.metrics import accuracy_score  # 정확도 계산 모듈

#### 모델 훈련용 데이터

nsmc_train_df = pd.read_csv("data/ratings_train.txt", encoding="utf-8", sep="\t", engine="python")
# data 폴더의 훈련용 데이터 파일 ratings_train.txt를 불러와 데이터프레임으로 저장

nsmc_train_df = nsmc_train_df[nsmc_train_df["document"].notnull()]
# document 내용이 null이 아닌 항목만 찾아서 다시 저장->결측치 제거

nsmc_train_df["document"] = nsmc_train_df["document"].apply(lambda x : re.sub(r'[^ㄱ-ㅎ|가-힣]+'," ", x))
# ㄱ~ㅎ으로 시작하거나 가~힣(모든 한글)의 모든 한글 문자를 제외한 나머지는 공백으로 치환->영어, 숫자, 특수문자 제거

#### 모델 평가용 데이터
nsmc_test_df = pd.read_csv("data/ratings_test.txt", encoding="utf-8", sep="\t", engine="python")
# data 폴더의 훈련용 데이터 파일 ratings_train.txt를 불러와 데이터프레임으로 저장

nsmc_test_df = nsmc_test_df[nsmc_test_df["document"].notnull()]
# document 내용이 null이 아닌 항목만 찾아서 다시 저장->결측치 제거

nsmc_test_df["document"] = nsmc_test_df["document"].apply(lambda x : re.sub(r'[^ㄱ-ㅎ|가-힣]+'," ", x))
# ㄱ~ㅎ으로 시작하거나 가~힣(모든 한글)의 모든 한글 문자를 제외한 나머지는 공백으로 치환->영어, 숫자, 특수문자 제거

okt = Okt()

def okt_tokenizer(text):  # 형태소 분석->형태소 단위로 토큰화 수행 함수
    tokens = okt.morphs(text)
    return tokens

tfidf = TfidfVectorizer(tokenizer=okt_tokenizer, min_df=3, max_df=0.9, ngram_range=(1,2), token_pattern=None)
tfidf.fit(nsmc_train_df["document"])
nsmc_train_tfidf = tfidf.transform(nsmc_train_df["document"])

SA_lr = LogisticRegression(random_state=0, max_iter=500)

SA_lr.fit(nsmc_train_tfidf, nsmc_train_df["label"])

params = {"C":[1,3,3.5,4,4.5,5]}  # 최적화 하이퍼 매개변수 후보군
SA_lr_grid_cv = GridSearchCV(SA_lr, param_grid=params, cv=3, scoring="accuracy", verbose=1)
# 최적화 하이퍼 파라미터를 찾아서 정확도가 제일 높은 최적 모델로 생성

SA_lr_grid_cv.fit(nsmc_train_tfidf, nsmc_train_df["label"])  # 설정값 조정
# 찾은 최적 하이퍼 파라미터 C 와 정확도를 반올림해서 소수점 4자리까지 출력

SA_lr_best = SA_lr_grid_cv.best_estimator_  # 최적 파라미터로 만든 best 모델 저장




with open("data/코로나_naver_news.json", encoding="utf-8") as j_f:
    data = json.load(j_f)  # json 파일 data로 저장

# print(data)

#  title, description 을 분리
data_title = []
data_description = []

for item in data:
    data_title.append(item["title"])  # 제목 문자열 들만 분리해서 리스트로 저장
    data_description.append(item["description"])  # 요약 문자열 들만 분리해서 리스트로 저장

# print(data_title)  # 1000개 title 문자열
# print(data_description)  # 1000개 description 문자열

### 제목, 요약 컬럼으로 dataframe 로 만들기
data_df = pd.DataFrame({"title":data_title, "description":data_description})
# print(data_df)

### 데이터(텍스트) 전처리
# title,description 컬럼에서 한글을 제외한 문자 제거
data_df["title"] = data_df["title"].apply(lambda x : re.sub(r"[^ㄱ-ㅎ|가-힝]+", " ", x))
data_df["description"] = data_df["description"].apply(lambda x : re.sub(r"[^ㄱ-ㅎ|가-힝]+", " ", x))
print(data_df)

data_title_tfidf = tfidf.transform(data_df["title"])  # title tf-idf 벡터
data_description_tfidf = tfidf.transform(data_df["description"])  # description tf-idf 벡터

data_title_predict = SA_lr_best.predict(data_title_tfidf)

data_df["title_label"] = data_title_predict  # 새로운 title_label 열 추가 (내용: 예측한 감성 결과 1 or 0)

data_description_predict = SA_lr_best.predict(data_description_tfidf)

data_df["description_label"] = data_description_predict  # 새로운 description_label 열 추가 (내용: 예측한 감성 결과 1 or 0)

data_df.to_csv("data/코로나뉴스_감성분석.csv", encoding="cp949") 
# data_df.to_csv("data/코로나뉴스_감성분석.csv", encoding="cp949", index=None)  # 로 작업해야 csv에 인덱스열이 안나옴
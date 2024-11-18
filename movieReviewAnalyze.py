import warnings
warnings.filterwarnings(action="ignore")  # 경고제거

import pandas as pd
import re
from konlpy.tag import  Okt  # 형태소 분석
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression  # 로지스틱 회귀모델
from sklearn.model_selection import GridSearchCV  # 하이퍼 매개변수 최적값 구하는 모듈
from sklearn.metrics import accuracy_score  # 정확도 계산 모듈

### 모델 훈련용 데이터

# data 폴더에 훈련용 데이터 불러와 dataframe 저장
nsmc_train_df = pd.read_csv("data/ratings_train.txt", encoding="utf-8", sep="\t", engine="python")

print(nsmc_train_df.info())

nsmc_train_df = nsmc_train_df[nsmc_train_df["document"].notnull()]
# document 내용이 null 이 아닌 항목만 찾아서 다시 저장 -> 결측치 제거

print(nsmc_train_df.info())

print(nsmc_train_df["label"].value_counts())  # label 칼럼의 빈도수 조사. 1, 0 이 각각 몇 개 인지


# nsmc_train_df["document"].apply(lambda x : re.sub(r"[^ㄱ-ㅎ|가-힝]+", " ", x))  한글만 남기고 제거 하는 방법
nsmc_train_df["document"] = nsmc_train_df["document"].apply(lambda x : re.sub(r"[^ㄱ-ㅎ|가-힝]+", " ", x))
#  ㄱ~ㅎ 으로 시작하거나 가~힣(모든 한글) 의 모든 한글문자를 제외한 나머지는 공백으로 치환. -> 영어, 숫자, 특수문자 제거
print(nsmc_train_df)

### 모델 평가용 데이터

# data 폴더에 훈련용 데이터 불러와 dataframe 저장
nsmc_test_df = pd.read_csv("data/ratings_test.txt", encoding="utf-8", sep="\t", engine="python")

print(nsmc_test_df.info())

nsmc_test_df = nsmc_test_df[nsmc_test_df["document"].notnull()]
# document 내용이 null 이 아닌 항목만 찾아서 다시 저장 -> 결측치 제거

print(nsmc_test_df.info())

print(nsmc_test_df["label"].value_counts())  # label 칼럼의 빈도수 조사. 1, 0 이 각각 몇 개 인지

nsmc_test_df["document"] = nsmc_test_df["document"].apply(lambda x : re.sub(r"[^ㄱ-ㅎ|가-힝]+", " ", x))
#  ㄱ~ㅎ 으로 시작하거나 가~힣(모든 한글) 의 모든 한글문자를 제외한 나머지는 공백으로 치환. -> 영어, 숫자, 특수문자 제거
print(nsmc_test_df)


okt = Okt()

def okt_tokenizer(text):  # 형태소 분석 - > 형태소 단위로 토큰화 수행 함수
    tokens = okt.morphs(text)
    return tokens

tfidf = TfidfVectorizer(tokenizer=okt_tokenizer, min_df=3, max_df=0.9, ngram_range=(1,2), token_pattern=None)
#  최소 몇번 이상 나온 것을 하겠냐 min_df=3
tfidf.fit(nsmc_train_df["document"])
nsmc_train_tfidf = tfidf.transform(nsmc_train_df["document"])
print(nsmc_train_tfidf)

# 로지스틱 회귀분석
SA_lr = LogisticRegression(random_state=0, max_iter=500)
SA_lr.fit(nsmc_train_tfidf, nsmc_train_df["label"])

params = {"C":[1,3,3.5,4,4.5,5]}  # 최적화 하이퍼 매개변수 후보군
SA_lr_grid_cv = GridSearchCV(SA_lr, param_grid=params, cv=3, scoring="accuracy", verbose=1)
## 최적화 하이퍼 파라미터를 찾아서 정확도가 제일 높은 최적 모델로 생성

SA_lr_grid_cv.fit(nsmc_train_tfidf, nsmc_train_df["label"])  # 설정값 조정

print(SA_lr_grid_cv.best_params_, round(SA_lr_grid_cv.best_score_,4))
# 찾은 최적 하이퍼 파라미터 C 와 정확도를 반올림해서 소스점 4자리까지 출력

SA_lr_best = SA_lr_grid_cv.best_estimator_  # 최적 파라미터로 만든 best 모델 저장

# 생선된 분석 모델 평가 -> 평가용 데이터를 tfidf 벡터로 변화 -> best.predict()
nsmc_test_tfidf = tfidf.transform(nsmc_test_df["document"])
nsmc_test_predict = SA_lr_best.predict(nsmc_test_tfidf)

nsmc_accuracy = accuracy_score(nsmc_test_df["label"], nsmc_test_predict)
print(f"감성 분석 모델 정확도 : {round(nsmc_accuracy, 4) * 100} %")

str = "웃자웃자~ 2024년 12월 시작~ 앗싸~! 좋은 예감! 난 잘될놈"
str = re.compile(r"[ㄱ-ㅎ|가-힝]+").findall(str)  # 한글이 아닌 문자 제거
str = [" ".join(str)]

print("========================================")

str_tfidt = tfidf.transform(str)
str_predict = SA_lr_best.predict(str_tfidt)
if (str_predict == 0):
    print(f"입력하신 {str} 문장은 부정 감성입니다!")
else:
    print(f"입력하신 {str} 문장은 긍정 감성입니다!")
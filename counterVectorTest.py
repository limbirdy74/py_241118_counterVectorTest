from nltk import word_tokenize
from collections import Counter  # 빈도수
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

text = ["I am a elementary school student", "And I am a boy"]  ## 문장이 2개

text_token_list = []

for str in text:
    text_token_list.append(word_tokenize(str))  # 문장 토큰화

print(text_token_list)
# [['I', 'am', 'a', 'elementary', 'school', 'student'], ['And', 'I', 'am', 'a', 'boy']]

str_counter = Counter()   #  파이썬은 신기해
for str in text_token_list:
    str_counter.update(str)  # 단어별 카운터를 딕셔너리로

print(str_counter)
# Counter({'I': 2, 'am': 2, 'a': 2, 'elementary': 1, 'school': 1, 'student': 1, 'And': 1, 'boy': 1})

text_bag = []

for key, value in str_counter.items():  #  중복 제거된 단어만 추출
    text_bag.append(key)

print(text_bag)
# ['I', 'am', 'a', 'elementary', 'school', 'student', 'And', 'boy']

text_count_vector = []

for str in text_token_list:
    str_vector = []
    for word in str:
        str_vector.append(str_counter[word])
    text_count_vector.append(str_vector)

print(text_count_vector)
# [[2, 2, 2, 1, 1, 1], [1, 2, 2, 2, 1]]

text = ["I am a elementary school student. And I am a boy"]
count_vector = CountVectorizer()
count_vector_array = count_vector.fit_transform(text).toarray()

print("------------------------")
print(count_vector_array)
# [[2 1 1 1 1 1]]

print(count_vector.vocabulary_)
#  {'am': 0, 'elementary': 3, 'school': 4, 'student': 5, 'and': 1, 'boy': 2}   # 인덱스. 단어의 갯수가 줄어들었음
#  I, a 불용어 제거

text = ["I am a great great elementary school student", "And I am a boy"] # 재정의
tfidfitm = TfidfVectorizer().fit(text)
tfidfitm_array = tfidfitm.transform(text).toarray()
print("------------------------")
print(tfidfitm_array)
print(tfidfitm.vocabulary_)

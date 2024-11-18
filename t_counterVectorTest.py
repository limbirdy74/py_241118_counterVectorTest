from nltk import word_tokenize
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer # count vector 생성 모듈

text = ["I am a elementary school student","And I am a boy"]  # 문장이 2개

text_token_list = []

for str in text:
    text_token_list.append(word_tokenize(str)) # 문장 토큰화

print(text_token_list)

str_counter = Counter()
for str in text_token_list:
    str_counter.update(str)  # 빈도수 도출

print(str_counter)

text_bag = []

for key, value in str_counter.items():
    text_bag.append(key)  # 중복 제거된 단어만 리스트에 저장

print(text_bag)

text_count_vector = []

for str in text_token_list:
    str_vector = []
    for word in str:
        str_vector.append(str_counter[word])
    text_count_vector.append(str_vector)

print(text_count_vector) # 카운트 벡터

text = ["I am a elementary school student. And I am a boy"]

count_vector = CountVectorizer()

count_vector_array = count_vector.fit_transform(text).toarray()
print("---------------------------------")

print(count_vector_array)

print(count_vector.vocabulary_)

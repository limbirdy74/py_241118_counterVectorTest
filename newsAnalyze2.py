import pandas as pd
from konlpy.tag import  Okt

data_df = pd.read_csv("data/코로나뉴스_감성분석.csv", encoding="cp949")
print(data_df.info())

print(data_df["title_label"].value_counts())
print(data_df["description_label"].value_counts())
# 위 빈도수 결과로 title, description 간 긍정과 부정의 분위기가 비슷하다고 판단됨
# title_label
# 0    602
# 1    398
# description_label
# 0    609
# 1    391

# 빈 dataframe 생성
NEG_data_df = pd.DataFrame(columns=["title", "title_label", "description", "description_label"])  # 부정감성
POS_data_df = pd.DataFrame(columns=["title", "title_label", "description", "description_label"])  # 긍정감성

# 긍정과 부정을 구분하여 각각의 dataframe 작성

for index, data in data_df.iterrows():
    title = data["title"]  # 1회전) 첫번쨰 행의 title 값
    description = data["description"]
    t_label = data["title_label"]
    d_label = data["description_label"]

    if d_label == 0:  # 부정 감성만 추출
        NEG_data_df = pd.concat([NEG_data_df, pd.DataFrame([[title, t_label, description, d_label]],
                                             columns=["title", "title_label", "description", "description_label"])],
                                             ignore_index=True)
    else:
        POS_data_df = pd.concat([POS_data_df, pd.DataFrame([[title, t_label, description, d_label]],
                                             columns=["title", "title_label", "description", "description_label"])],
                                             ignore_index=True)

NEG_data_df.to_csv("data/코로나뉴스_NEGATIVE.csv", encoding="euc-kr")
POS_data_df.to_csv("data/코로나뉴스_POSITIVE.csv", encoding="euc-kr")

print(f"부정 감성 요약 뉴스 개수 :{len(NEG_data_df)}")
print(f"긍정 감성 요약 뉴스 개수 :{len(POS_data_df)}")

POS_description = POS_data_df["description"]  # 긍정 요약 뉴스에서 description 컬럼만 추출
POS_description_noun_tk = [] # 명사만 추출하여 담길 빈 리스트

okt = Okt()

for des in POS_description:
    POS_description_noun_tk.append(okt.nouns(des))

print(POS_description_noun_tk)
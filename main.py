import os
import tarfile
import urllib
import urllib.request
import matplotlib.pyplot as plt

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOSUING_PATH = os.path.join("dataset","housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url = HOUSING_URL, housing_path=HOSUING_PATH):
    os.makedirs(housing_path,exist_ok=True)
    tgz_path = os.path.join(housing_path,"housing.tgz")
    urllib.request.urlretrieve(housing_url,tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()




import pandas as pd
import numpy as np

def load_housing_data(housing_path = HOSUING_PATH):
    csv_path = os.path.join(housing_path,"housing.csv")
    return pd.read_csv(csv_path)

housing = load_housing_data()


#테스트 데이터 생성하기

def split_train_test(data : pd.DataFrame, test_ratio):
    # 내부 인덱스 요소를 섞는다. 
    shuffled_indices = np.random.permutation(len(data))
    # 테스트 셋 사이즈 비율만큼 길이를 정함
    test_set_size = int(len(data) * test_ratio)
    # 데이터를 비율 만큼 나눈다.
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    # iloc이 인덱스 참조라 인덱스를 기준으로 테스트 데이터와 평가 데이터를 리턴한다.
    return data.iloc[train_indices],data.iloc[test_indices]


train_set, test_set = split_train_test(housing, 0.2)

from zlib import crc32


# 데이터가 추가되거나 순서가 바뀌어도 일관성을 유지할 수 있다.
def test_set_check(identifier, test_ratio):
    # 0xffffffff 로 연산하는 것은 32bit 정수로 맞춰주기 위함
    # 리턴값은 2*32 * 비율에 포함하는 것이랑 안하는 것이랑 true, false 값으로 리턴한다.
    # 해쉬값을 기준으로 하기 떄문에 순서가 바뀔 이유가 없고 항상 동일한 결과를 얻을 수 있다.
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_radio, id_col):
    ids = data[id_col]
    # True, False 형식의 데이터프레임으로 반환하여줌줌
    in_test_set = ids.apply(lambda id_ : test_set_check(id_,test_radio))
    # in_test_set의 index 행을 돌면서 True or False면 각각의 데이터의 나눠서 리턴함
    return data.loc[~in_test_set], data.loc[in_test_set]

# 인덱스 리셋
housing_with_id = housing.reset_index()


train_set, test_set = split_train_test_by_id(housing_with_id,0.2,"index")

#위경도를 합쳐서 임의의 아이디 생성, 특별한 식별자가 없음
housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id,0.2,"id")

# sklearn으로 데이터를 여러 서브셋으로 나누는 방법

from sklearn.model_selection import train_test_split

# 삽입할 데이터, 테스트 데이터의 비율, 난수값 -> 난수값 고정시 같은 데이터 출력력
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

# 데이터의 편향을 발생하지 않게 하기 위한 카테고리 분류
# 카테고리 1은 0 ~ 1.5, 카테고리 2는 1.5 ~ 3 ... 이런식으로 분류
# bins 사이의 위치한 값들은 labels의 카테고리 값으로 변경됨
housing["income_cat"] = pd.cut(housing["median_income"],
                               #bins의 값은 순차적으로 올라가야한다
                               bins=[0. ,1.5 ,3.0 ,4.5 ,6. ,np.inf],
                               labels=[1,2,3,4,5])

# print(housing["income_cat"].hist())
#소득카테고리 기반으로 계층 샘플링

from sklearn.model_selection import StratifiedShuffleSplit
# 데이터를 나눌 횟수 = 1, 테스트 비율 = 0.2, 랜덤시드 = 42
# StratifiedShuffleSplit 함수는 계층적 샘플링이 가능하도록 해준다.
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, tes_idx in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_idx]
    strat_test_set = housing.loc[tes_idx]

# 훈련셋에서 위에서 카테고리로 분류한 데이터랑 비율이 비슷하다.
# print(strat_train_set["income_cat"].value_counts()/len(strat_train_set))

#income_cat 특성을 삭제하여 원상태로 돌림

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

housing = strat_train_set.copy()

# scatter : 산점도, alpha를 0.1로 주면 데이터 포인트가 밀집된 지역을 보여줌
# alpha : 불투명하게 해주는 옵션인데 흐려지면서 겹치는 부분이 더 잘보이는 것이다.
# figsize : 그래픽의 크기 인치로
# c : 점마커의 색상을 값에 따라 변하게 함
# cmap : 색상지도, 파-초-노-빨 순으로 변함
# colorbar : 옆에 색상바를 표시
# sharex : 플롯을 여러 서브플롯으로 나눌 떄 X축을 공유할지의 대한 값
# 아직 완벽 이해는 안감...
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"]/100, label="population", figsize=(10,7),
             c="median_house_value",cmap=plt.get_cmap("jet"), colorbar=True,
             sharex=False)

#데이터 셋이 크지않아 corr() 이라는 함수를 사용하여 모든 특성 간의 표준 상관계수을 계산할 수 있다
# 중간에 문자열이 있어 오류가 발생하였고 자동적으로 숫자열의 상관관계를 분석하는 numeric_only=True 를 사용
corr_matrix = housing.corr(numeric_only=True)

print(corr_matrix["median_house_value"].sort_values(ascending=False))

# 94p까지 진행 완료
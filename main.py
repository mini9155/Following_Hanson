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

#가상환경 active 실행이 안될 시


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
# housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
#              s=housing["population"]/100, label="population", figsize=(10,7),
#              c="median_house_value",cmap=plt.get_cmap("jet"), colorbar=True,
#              sharex=False)

#데이터 셋이 크지않아 corr() 이라는 함수를 사용하여 모든 특성 간의 표준 상관계수을 계산할 수 있다
# 중간에 문자열이 있어 오류가 발생하였고 자동적으로 숫자열의 상관관계를 분석하는 numeric_only=True 를 사용
corr_matrix = housing.corr(numeric_only=True)

# print(corr_matrix["median_house_value"].sort_values(ascending=False))

# 94p까지 진행 완료
# 
from pandas.plotting import scatter_matrix

# atrtibute = ["median_house_value", "median_income", "total_rooms",
#              "housing_median_age"]

# scatter_matrix(housing[atrtibute], figsize=(12,8))

# housing.plot(kind="scatter", x="median_income", y="median_house_value",
#              alpha=0.1)

housing["room_per_househlod"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_househlod"] = housing["population"]/housing["households"]

# 상관관계 확인 함수
corr_matrix = housing.corr(numeric_only=True)
# print(corr_matrix["median_house_value"].sort_values(ascending=False))

""" Result:
median_house_value          1.000000
median_income               0.687151
room_per_househlod          0.146255
total_rooms                 0.135140
housing_median_age          0.114146
households                  0.064590
total_bedrooms              0.047781
population_per_househlod   -0.021991
population                 -0.026882
longitude                  -0.047466
latitude                   -0.142673
bedrooms_per_room          -0.259952
"""


"""
# 머신러닝 알고리즘을 위한 데이터 준비
## 데이터를 수동이 아닌 함수를 만들어 자동화해야 하는 이유
1. 어떤 데이터셋에 대해서도 데이터 변환을 손쉽게 반복 가능
2. 향후 프로젝트에서 사용할 수 있는 변환 라이브러리를 점진적으로 구축하게 됨
3. 실제 시스템에서 알고리즘에 새 데이터를 주입하기 전에 변환시키는데 이 함수를 사용 가능
4. 여러가지 데이터 변환을 쉽게 시도해볼 수 있고 어떤 조합이 가장 좋은지 확인하는 데 편리함.

median_house_value 열 제거

"""

housing = strat_train_set.drop("median_house_value",axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

"""
# 데이터 정제
특성값이 없을 경우 방법
1. 해당 구역 제거
2. 전체 특성 삭제
3. 임의 값 삽임, 0, 평균, 중간값 등 Ex) dropna, fillna, drop
"""

from sklearn.impute import SimpleImputer
#누락된 값은 중간값으로 바꿔주기 위한 인스턴스 생성
imputer = SimpleImputer(strategy="median")
# ocen_proximity 열을 제거
housing_num = housing.drop("ocean_proximity",axis=1)
# 중앙값을 계산하게 위해 데이터를 채우기 위한 작업 시작
imputer.fit(housing_num)

# imputer는 각 특성의 중간값을 계산해서 그 결과를 객체의 statistics_ 속성에 저장.
# print(imputer.statistics_)

X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index=housing_num.index)

housing_cat = housing[["ocean_proximity"]]
# print(housing_cat.head(10))

# 머신러닝 알고리즘은 텍스트가 아닌 숫자를 다루기 위한 변환 클래스

# OrdinalEncoder 순서가 있는 인코딩 방식
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
housing_cat_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)

# print(housing_cat_encoded[:10])
# print(ordinal_encoder.categories_)

from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
# 다차원 희소 행렬을 2차원 배열로 출력
# print(housing_cat_1hot.toarray())
# print(cat_encoder.categories_)
# 106p

from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

# BaseEstimator 은 sklearn 의 추정기 표준을 따르기 위해 사용
# TransformerMixin 은 .fit() 과 .transform() 매서드릴 가진 반환기를 구성하기 쉽게 해준다.
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:,household_ix]
        population_per_household = X[:,population_ix] / X[:,household_ix]
        if(self.add_bedrooms_per_room):
            bedrooms_per_room = X[:,bedrooms_ix] / X[:,rooms_ix]
            # C_는 새로운 열을 X에 추가
            return np.c_[X,rooms_per_household, population_per_household, bedrooms_per_room]
        
        else:
            return np.c_[X,rooms_per_household, population_per_household]
        
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_atrribs = attr_adder.transform(housing.values)

#_get_params, set_params 함수는 사이킷런의 파이프라인과 그리드 탐색에 꼭 필요한 메서드이다.
# 모든 추정기 변환기는 BaseEstimator을 상속해야함.

"""
전체방의 개수는 6 ~ 39.320인 반면 중간 소득의 범위는 0 에서 15이다.
모든 특성의 범위를 같게 만들어 줘야 한다.
주로 min-max 스케일링과 표준화가 널리 사용된다.

min-max 스케일링
- 데이터에서 최소값을 뺀 후 최댓값과 최솟값의 차이로 나눈다.
- 해당하는 함수는 MinMaxScaler, 만약 0 ~ 1 사이를 원하지 않는다면 feature_range 매개변수로 범위를 정할 수 있다.

표준화
- 먼저 평균을 뺀 후 표준편차로 나누어 결과 분포의 분산이 1이 되도록 함.
"""

# 변환 파이프라인

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    # 결측값을 채우는 변환기, 중앙값으로 채움
    ("imputer",SimpleImputer(strategy="median")),
    # 사용자 정의 변환기
    ("attribs_adder",CombinedAttributesAdder()),
    # 데이터를 정규화함
    ("std_scaler",StandardScaler()),
])

housing_num_tr = num_pipeline.fit_transform(housing_num)

from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
    ("num",num_pipeline, num_attribs),
    ("cat",OneHotEncoder(), cat_attribs)
])

housing_prepared = full_pipeline.fit_transform(housing)

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]

some_data_prepared = full_pipeline.transform(some_data)
# print(f"예측 : {lin_reg.predict(some_data_prepared)}")
# print(f"레이블 : {list(some_labels)}")

from sklearn.metrics import mean_squared_error
# # 전처리된 데이터 예측
# housing_predictions = lin_reg.predict(housing_prepared)
# # MSE 평균 제곱근 오차 계산
# lin_mse = mean_squared_error(housing_labels, housing_predictions)
# # RMSE 평균 제곱근 오차를 구하는 sqrt 함수(제곱근 생성)
# lin_rmse = np.sqrt(lin_mse)


# from sklearn.tree import DecisionTreeRegressor

# tree_reg = DecisionTreeRegressor()
# tree_reg.fit(housing_prepared, housing_labels)
# housing_predictions = tree_reg.predict(housing_prepared)
# tree_mse = mean_squared_error(housing_labels, housing_predictions)
# tree_rmse = np.sqrt(tree_mse)

from sklearn.model_selection import cross_val_score

# scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
#                          scoring="neg_mean_squared_error",cv=10)
# tree_rmse_scores = np.sqrt(-scores)


def display_scores(scores):
    print("점수 : ", scores)
    print("평균 : ", scores.mean())
    print("표준편차 : ", scores.std())

# display_scores(scores=tree_rmse_scores)

# lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
#                              scoring="neg_mean_squared_error", cv=10)
# lin_rmse_scores = np.sqrt(-lin_scores)
# # display_scores(lin_rmse_scores)

from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)
housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
display_scores(forest_rmse)
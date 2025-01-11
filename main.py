import os
import tarfile
import urllib
import urllib.request


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
    in_test_set = ids.apply(lambda id_ : test_set_check(id_,test_radio))
    return data.loc[~in_test_set], data.loc[in_test_set]

# 인덱스 리셋셋
housing_with_id = housing.reset_index()

# 열의 이름이 "index" 인 True, False 값의 데이터 프레임이 반환됨됨
train_set, test_set = split_train_test_by_id(housing_with_id,0.2,"index")

print(train_set)
print(test_set)
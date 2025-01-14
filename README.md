# Follwing Hans_on
learning machine_learing with hanson

#### 가상환경 활성화가 안될 시

 '''
 Get-ExecutionPolicy

 만약 출력이 : Restricted 이면 스크립트 실행 허용 X

 Set-ExecutionPolicy RemoteSigned (로컬 스크립트는 실행 가능)

 
 ''' 

#### 계층적 샘플링 - stratified smapling
- 특정 비율에 맞춘 샘플링, 만약 여론조사시 인구의 40퍼가 여성이고, 인구의 60퍼가 남성이라면 조사시 이 비율을 맞추는 것을 의미한다


#### 사이킷런의 설계 철학

1. 일관성 : 모든 객체가 일관되고 단순환 인터페이스를 공유
 - 추정기 (estimator)
   * 데이터셋을 기반으로 일련의 모델 파라미터들을 추정하는 객체이며 추정은 fit() 수행되고, 추정 과정에서 필요한 매개변수는 하이퍼파라미터로 간주되고 인스턴스 변수로 저장됩니다.
 - 변환기 (transformer)
   * 데이터셋을 변환하는 추정기는 변환기라고 함. 변환은 transform 이 수행, fut_transform 이라는 함수도 있음.
 - 예측기 (perdictor)
   * 일부 추정기는 주어진 데이터셋에 대한 예측기를 만들 수 있음. predict 함수로 예측값을 받아 반환, score 함수로 평가
2. 검사기능
   - 모든 추정기의 하이퍼파라미터는 인스턴스 변수로 확인 가능
3. 클래스 남용 방지 




import pandas as pd
import numpy as np

# df1: big_data_set1_f.csv 파일 불러오기
df1 = pd.read_csv('big_data_set1_f.csv', encoding='cp949')
print("df1 기본 정보:")
df1.info()
print("\ndf1 상위 5행:")
print(df1.head())

# df2: big_data_set2_f.csv 파일 불러오기
df2 = pd.read_csv('big_data_set2_f.csv', encoding='cp949')
print("df2 기본 정보:")
df2.info()
print("\ndf2 상위 5행:")
print(df2.head())

# df3: big_data_set3_f.csv 파일 불러오기
df3 = pd.read_csv('big_data_set3_f.csv', encoding='cp949')
print("df3 기본 정보:")
df3.info()
print("\ndf3 상위 5행:")
print(df3.head())

print("\n=== 모든 CSV 파일 불러오기 완료 ===")

# 데이터 정제 및 Target 변수 생성
print("\n=== 데이터 정제 및 Target 변수 생성 시작 ===")

# 1. 데이터 정제
print("\n1. 특수값 처리 중...")
# df2와 df3에서 -999999.9 값을 np.nan으로 변환
df2 = df2.replace(-999999.9, np.nan)
df3 = df3.replace(-999999.9, np.nan)
print("df2, df3의 -999999.9 값을 np.nan으로 변환 완료")

# 2. 날짜 컬럼 타입 변경
print("\n2. 날짜 컬럼 타입 변경 중...")
# df1의 날짜 컬럼들
df1['ARE_D'] = pd.to_datetime(df1['ARE_D'], format='%Y%m%d')
df1['MCT_ME_D'] = pd.to_datetime(df1['MCT_ME_D'], format='%Y%m%d')
print("df1의 'ARE_D', 'MCT_ME_D' 컬럼을 datetime으로 변환 완료")

# df2와 df3의 날짜 컬럼들
df2['TA_YM'] = pd.to_datetime(df2['TA_YM'], format='%Y%m')
df3['TA_YM'] = pd.to_datetime(df3['TA_YM'], format='%Y%m')
print("df2, df3의 'TA_YM' 컬럼을 datetime으로 변환 완료")

# 3. Target 변수 생성
print("\n3. Target 변수 'is_closed' 생성 중...")
# MCT_ME_D에 날짜가 있으면(폐업) 1, 결측치면(정상 운영) 0
df1['is_closed'] = df1['MCT_ME_D'].notna().astype(int)
print("Target 변수 'is_closed' 생성 완료")

# 4. 결과 확인
print("\n=== Target 변수 분포 확인 ===")
print("폐업/정상 가맹점 분포:")
print(df1['is_closed'].value_counts())
print("\n분포 비율:")
print(df1['is_closed'].value_counts(normalize=True))

print("\n=== 데이터 정제 및 Target 변수 생성 완료 ===")

# 데이터프레임 병합
print("\n=== 데이터프레임 병합 시작 ===")

# 1단계: df2와 df3 병합 (월별 이용 정보 + 월별 고객 정보)
print("\n1단계: df2(월별 이용 정보)와 df3(월별 고객 정보) 병합 중...")
print(f"df2 크기: {df2.shape}")
print(f"df3 크기: {df3.shape}")

merged_df = pd.merge(df2, df3, on=['ENCODED_MCT', 'TA_YM'], how='inner')
print(f"1단계 병합 완료 - merged_df 크기: {merged_df.shape}")

# 2단계: merged_df와 df1 병합 (가맹점 개요 정보 추가)
print("\n2단계: merged_df와 df1(가맹점 개요 정보) 병합 중...")
print(f"merged_df 크기: {merged_df.shape}")
print(f"df1 크기: {df1.shape}")

final_df = pd.merge(merged_df, df1, on='ENCODED_MCT', how='left')
print(f"2단계 병합 완료 - final_df 크기: {final_df.shape}")

# 결과 확인
print("\n=== 최종 병합 결과 확인 ===")
print(f"최종 데이터프레임 크기: {final_df.shape}")
print(f"컬럼 수: {final_df.shape[1]}, 행 수: {final_df.shape[0]}")

print("\n최종 데이터프레임 상위 5행:")
print(final_df.head())
final_df.to_csv('final_df.csv', index=False, encoding='utf-8-sig')
print("final_df를 'final_df.csv' 파일로 저장 완료")


print("\n=== 데이터프레임 병합 완료 ===")

# 파생 변수(피처) 생성
print("\n=== 파생 변수 생성 시작 ===")

# 1. 운영 개월 수 계산
print("\n1. 운영 개월 수 계산 중...")
# TA_YM과 ARE_D의 차이를 월 단위로 계산
final_df['operating_months'] = ((final_df['TA_YM'].dt.year - final_df['ARE_D'].dt.year) * 12 + 
                                (final_df['TA_YM'].dt.month - final_df['ARE_D'].dt.month))
print("operating_months 컬럼 생성 완료")

# 2. 순위 컬럼 전처리
print("\n2. 순위 컬럼 전처리 중...")

# RC_M1_SAA에서 맨 앞 숫자만 추출하여 sales_rank_num 생성
final_df['sales_rank_num'] = final_df['RC_M1_SAA'].str.extract(r'^(\d+)').astype(int)
print("sales_rank_num 컬럼 생성 완료")

# RC_M1_UE_CUS_CN에서 맨 앞 숫자만 추출하여 customer_rank_num 생성
final_df['customer_rank_num'] = final_df['RC_M1_UE_CUS_CN'].str.extract(r'^(\d+)').astype(int)
print("customer_rank_num 컬럼 생성 완료")

# 3. 직전 월 대비 주요 지표 변화량 계산
print("\n3. 직전 월 대비 주요 지표 변화량 계산 중...")

# 각 가맹점별로 그룹화하여 직전 월 대비 차이 계산
final_df = final_df.sort_values(['ENCODED_MCT', 'TA_YM'])

# 숫자 랭크 컬럼을 사용하여 차이 계산
final_df['sales_rank_diff'] = final_df.groupby('ENCODED_MCT')['sales_rank_num'].diff()
final_df['customer_count_rank_diff'] = final_df.groupby('ENCODED_MCT')['customer_rank_num'].diff()
print("sales_rank_diff, customer_count_rank_diff 컬럼 생성 완료")

# 4. 결과 확인
print("\n4. 결과 확인...")
print("\n=== 순위 컬럼 전처리 및 차이 계산 결과 ===")
print("원본 컬럼과 새로 생성된 숫자 및 차이 컬럼 상위 10개 행:")
result_columns = ['RC_M1_SAA', 'sales_rank_num', 'sales_rank_diff']
print(final_df[result_columns].head(10))

print("\n=== 추가 확인: 고객 수 순위 관련 컬럼 ===")
customer_columns = ['RC_M1_UE_CUS_CN', 'customer_rank_num', 'customer_count_rank_diff']
print(final_df[customer_columns].head(10))

print("\n=== 순위 컬럼 전처리 및 차이 계산 완료 ===")


final_df.to_csv('final_df1.csv', index=False, encoding='utf-8-sig')

# 모델링을 위한 최종 데이터 준비
print("\n=== 모델링을 위한 최종 데이터 준비 시작 ===")

# 1. 피처(X)와 타겟(y) 정의
print("\n1. 피처(X)와 타겟(y) 정의 중...")

# 타겟 변수 정의
y = final_df['is_closed']
print(f"타겟 변수 'y' 생성 완료 - shape: {y.shape}")

# 피처 변수 정의 (요청된 컬럼들)
feature_columns = ['operating_months', 'sales_rank_num', 'customer_rank_num', 
                   'sales_rank_diff', 'customer_count_rank_diff', 'MCT_UE_CLN_REU_RAT']
X = final_df[feature_columns]
print(f"피처 변수 'X' 생성 완료 - shape: {X.shape}")
print(f"사용된 피처 컬럼: {feature_columns}")

# 2. 결측치(NaN) 처리
print("\n2. 결측치(NaN) 처리 중...")
print("처리 전 결측치 개수:")
print(X.isnull().sum())

# 각 컬럼의 중앙값으로 결측치 채우기
X = X.fillna(X.median())
print("처리 후 결측치 개수:")
print(X.isnull().sum())
print("결측치 처리 완료")

# 3. 학습용/테스트용 데이터 분리
print("\n3. 학습용/테스트용 데이터 분리 중...")

# scikit-learn의 train_test_split 임포트
from sklearn.model_selection import train_test_split

# 80:20 비율로 데이터 분리 (stratify 옵션으로 비율 유지)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)
print("데이터 분리 완료")

# 4. 결과 확인
print("\n4. 결과 확인...")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train 분포:")
print(y_train.value_counts(normalize=True))
print(f"y_test 분포:")
print(y_test.value_counts(normalize=True))

print("\n=== 모델링을 위한 최종 데이터 준비 완료 ===")

# 베이스라인 모델 생성 및 평가
print("\n=== 베이스라인 모델 생성 및 평가 시작 ===")

# 1. 모델 학습
print("\n1. 로지스틱 회귀 모델 학습 중...")

# scikit-learn에서 LogisticRegression 임포트
from sklearn.linear_model import LogisticRegression

# 로지스틱 회귀 모델 생성 및 학습
logistic_model = LogisticRegression(random_state=42, max_iter=1000)
logistic_model.fit(X_train, y_train)
print("로지스틱 회귀 모델 학습 완료")

# 2. 예측 수행
print("\n2. 테스트 데이터 예측 중...")
y_pred = logistic_model.predict(X_test)
print("예측 완료")

# 3. 모델 평가
print("\n3. 모델 평가 중...")

# scikit-learn의 metrics에서 필요한 함수들 임포트
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 정확도(Accuracy) 계산 및 출력
accuracy = accuracy_score(y_test, y_pred)
print(f"\n=== 모델 평가 결과 ===")
print(f"정확도(Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)")

# 혼동 행렬(Confusion Matrix) 계산 및 출력
print(f"\n혼동 행렬(Confusion Matrix):")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# 분류 리포트(Classification Report) 계산 및 출력
print(f"\n분류 리포트(Classification Report):")
report = classification_report(y_test, y_pred)
print(report)

# 4. 결과 해석 주석
print(f"\n=== 결과 해석 ===")
print("""
혼동 행렬(Confusion Matrix) 해석:
- [0,0]: 실제 정상(0)이고 예측도 정상(0)인 경우 (True Negative)
- [0,1]: 실제 정상(0)이지만 예측은 폐업(1)인 경우 (False Positive) 
- [1,0]: 실제 폐업(1)이지만 예측은 정상(0)인 경우 (False Negative)
- [1,1]: 실제 폐업(1)이고 예측도 폐업(1)인 경우 (True Positive)

분류 리포트(Classification Report) 해석:
- Precision (정밀도): 예측한 폐업 중 실제 폐업인 비율
- Recall (재현율): 실제 폐업 중 올바르게 예측한 비율  
- F1-score: 정밀도와 재현율의 조화평균
- Support: 각 클래스의 실제 샘플 수
- Macro avg: 각 클래스별 지표의 평균
- Weighted avg: 각 클래스의 샘플 수에 따른 가중평균
""")

print("\n=== 베이스라인 모델 생성 및 평가 완료 ===")



# class_weight='balanced'를 적용한 로지스틱 회귀 모델 재학습 및 평가
print("\n=== 가중치 조정(balanced) 로지스틱 회귀 재학습 및 평가 시작 ===")

# 1. 가중치 조정한 모델 생성 및 학습
print("\n1. balanced 로지스틱 회귀 모델 학습 중...")
balanced_model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
balanced_model.fit(X_train, y_train)
print("balanced 로지스틱 회귀 모델 학습 완료")

# 2. 예측 수행
print("\n2. balanced 모델로 테스트 데이터 예측 중...")
y_pred_balanced = balanced_model.predict(X_test)
print("예측 완료")

# 3. 평가 (혼동 행렬, 분류 리포트)
print("\n3. balanced 모델 평가 중...")
print(f"\n혼동 행렬(Confusion Matrix) - balanced:")
cm_bal = confusion_matrix(y_test, y_pred_balanced)
print(cm_bal)

print(f"\n분류 리포트(Classification Report) - balanced:")
report_bal = classification_report(y_test, y_pred_balanced)
print(report_bal)

print("\n=== 가중치 조정(balanced) 로지스틱 회귀 재학습 및 평가 완료 ===")

# LightGBM 모델 학습 및 평가
print("\n=== LightGBM 모델 학습 및 평가 시작 ===")

# 1. LightGBM 모델 학습
print("\n1. LightGBM 모델 학습 중...")

# lightgbm 라이브러리에서 LGBMClassifier 임포트
from lightgbm import LGBMClassifier

# LGBMClassifier 모델 생성 (is_unbalance=True로 불균형 데이터 처리)
lgb_model = LGBMClassifier(random_state=42, is_unbalance=True, verbose=-1)
lgb_model.fit(X_train, y_train)
print("LightGBM 모델 학습 완료")

# 2. 예측 및 평가
print("\n2. LightGBM 모델로 테스트 데이터 예측 중...")
y_pred_lgb = lgb_model.predict(X_test)
print("예측 완료")

# 3. 평가 (혼동 행렬, 분류 리포트)
print("\n3. LightGBM 모델 평가 중...")
print(f"\n혼동 행렬(Confusion Matrix) - LightGBM:")
cm_lgb = confusion_matrix(y_test, y_pred_lgb)
print(cm_lgb)

print(f"\n분류 리포트(Classification Report) - LightGBM:")
report_lgb = classification_report(y_test, y_pred_lgb)
print(report_lgb)

print("\n=== LightGBM 모델 학습 및 평가 완료 ===")

# 피처 중요도 분석
print("\n=== 피처 중요도 분석 시작 ===")

# 1. 피처 중요도 계산
print("\n1. 피처 중요도 계산 중...")
feature_importance = lgb_model.feature_importances_
print("피처 중요도 계산 완료")

# 피처 이름과 중요도를 매핑
feature_names = feature_columns
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
})

# 중요도 순으로 정렬
importance_df = importance_df.sort_values('importance', ascending=True)
print("\n피처 중요도 (낮은 순):")
print(importance_df)

# 2. 피처 중요도 시각화
print("\n2. 피처 중요도 시각화 중...")

# matplotlib과 seaborn 임포트
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정 (Windows 환경)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 그래프 크기 설정
plt.figure(figsize=(10, 6))

# 가로 막대 그래프 생성
plt.barh(importance_df['feature'], importance_df['importance'], color='skyblue', alpha=0.7)

# 그래프 제목과 라벨 설정
plt.title('LightGBM 모델 피처 중요도', fontsize=16, fontweight='bold')
plt.xlabel('중요도 점수', fontsize=12)
plt.ylabel('피처', fontsize=12)

# 격자 추가
plt.grid(axis='x', alpha=0.3)

# 중요도 값 표시
for i, v in enumerate(importance_df['importance']):
    plt.text(v + 0.1, i, f'{v:.2f}', va='center', fontsize=10)

# 레이아웃 조정
plt.tight_layout()

# 그래프 저장 및 표시
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

print("피처 중요도 그래프 생성 완료 (feature_importance.png로 저장됨)")

# 3. 결과 해석
print("\n3. 피처 중요도 해석:")
print("=" * 50)
for idx, row in importance_df.iterrows():
    print(f"{row['feature']}: {row['importance']:.4f}")

print("\n=== 피처 중요도 분석 완료 ===")

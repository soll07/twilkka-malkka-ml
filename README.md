# 🎬 튈까말까 — OTT 고객 이탈 예측

>  Netflix 사용자 행동 데이터를 기반으로 OTT 서비스 고객 이탈을 예측하는 머신러닝 프로젝트

---

## 📌 프로젝트 개요

OTT 서비스 시장의 경쟁이 심화되면서 신규 고객 확보보다 **기존 고객 유지**가 더 중요한 과제로 부상하고 있습니다. 일반적으로 신규 고객 확보 비용은 기존 고객 유지 비용보다 평균 5배 이상 높은 것으로 알려져 있습니다.

본 프로젝트는 **Push-Pull-Mooring(PPM) 이론**을 분석 배경으로 삼아 Netflix 사용자 데이터를 분석하고, 이탈 가능성이 높은 고객을 사전에 식별하는 머신러닝 기반 예측 모델을 구축합니다. 구축된 모델은 Streamlit 대시보드로 시각화하여 CRM 실무에 활용할 수 있도록 설계하였습니다.

> **PPM 모델이란?** 이용자가 서비스를 떠나는 이유를 세 가지로 설명하는 이론입니다. 현재 서비스의 불만족(Push) + 경쟁 서비스의 매력(Pull) + 전환 비용·습관 등 이탈을 억제하는 요소(Mooring)가 복합적으로 작용해 이탈이 발생합니다.

```
1차 데이터(설문) ──┐
                   ├──▶ EDA · 전처리 · 피처 엔지니어링 ──▶ ML 모델링 ──▶ Streamlit 대시보드
2차 데이터(행동) ──┘
```

---

## 👥 팀원

|                                                                  김민하                                                                   |                                                                                      장한재                                                                                      |                                                                 배재현                                                                  |                                                                  전윤하                                                                   |                                                               정다솔                                                                |                                                                                         이창우                                                                                          |
|:--------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|     <img src="https://m.partydeyuni.co.kr/web/product/big/202310/3670b31b7884accc1690b5ac77132081.jpg" width="150" height="150"/>      |            <img src="https://i.namu.wiki/i/FkXY7nH7Bu5JNiZLch_10eMZ1HRgjUcCbMzvceMs7qgOtSUOUgpkYRTVkRPJsb_zaDxDaE4DEE2LR7dTX-7ZOQ.webp" width="150" height="150"/>            | <img src="https://img.khan.co.kr/lady/2023/07/25/news-p.v1.20230725.e1c02b0bdeb04153bc35d5f1e7c58952.png" width="150" height="150"/> |             <img src="https://ojsfile.ohmynews.com/STD_IMG_FILE/2022/0127/IE002931855_STD.jpg" width="150" height="150"/>              |      <img src="https://img.insight.co.kr/static/2021/10/06/700/img_20211006190524_f59v29t4.webp" width="150" height="150"/>      |               <img src="https://i.namu.wiki/i/3knMp23cDOe3hSQaHfvNrlTewKi8nGPz1Dg89uDwP6VKadSI2iUi9ltEQkUjSJvAAusl0XHjSFywWJpvt7ISWA.webp" width="150" height="150"/>                |
| [![GitHub](https://img.shields.io/badge/GitHub-leedhroxx-181717?style=flat&logo=github&logoColor=white)](https://github.com/leedhroxx) |                     [![GitHub](https://img.shields.io/badge/GitHub-rusidian-181717?style=flat&logo=github&logoColor=white)](https://github.com/rusidian)                      | [![GitHub](https://img.shields.io/badge/GitHub-rshyun24-181717?style=flat&logo=github&logoColor=white)](https://github.com/rshyun24) | [![GitHub](https://img.shields.io/badge/GitHub-yoonha315-181717?style=flat&logo=github&logoColor=white)](https://github.com/yoonha315) | [![GitHub](https://img.shields.io/badge/GitHub-soll07-181717?style=flat&logo=github&logoColor=white)](https://github.com/soll07) |                         [![GitHub](https://img.shields.io/badge/GitHub-Gloveman-181717?style=flat&logo=github&logoColor=white)](https://github.com/Gloveman)                         |
|                                         Logistic Regression 모델링<br/>EDA 수행<br/>회의록 및 README 작성                                         |                                                                    데이터 수집<br/>EDA 수행<br/>Streamlit 대시보드 구현                                                                    |                                            데이터수집 및 전처리<br/>PPT 작성<br/>기획 및 발표<br/>EDA 수행                                             |                                           GitHub 총괄 관리<br/>Random Forest, XGBoost<br/>EDA 수행                                           |                                  Streamlit 구현 및 UI 디자인(Figma)<br/>EDA 수행<br/>README 작성<br/>노션정리                                  |                                                            데이터 수집 및 전처리<br/>feature 생성<br/>train 및 predict 파이프라인 구축 및 모듈화                                                            |

---
## 📄 WBS

```mermaid
flowchart LR
    A[데이터 수집]:::blueNode --> B[데이터 탐색]:::greenNode
    B --> C[정제 및 전처리]:::yellowNode
    C --> D[피처 엔지니어링]:::redNode
    D --> E[모델 학습 및 평가]:::purpleNode
    E --> F[Streamlit 대시보드]:::orangeNode

    %% 노드 스타일 정의
    classDef blueNode fill:#cce5ff,stroke:#3399ff,stroke-width:2px;
    classDef greenNode fill:#d4edda,stroke:#28a745,stroke-width:2px;
    classDef yellowNode fill:#fff3cd,stroke:#ffc107,stroke-width:2px;
    classDef redNode fill:#f8d7da,stroke:#dc3545,stroke-width:2px;
    classDef purpleNode fill:#e2d6f9,stroke:#6f42c1,stroke-width:2px;
    classDef orangeNode fill:#ffe5b4,stroke:#ff851b,stroke-width:2px;
    
```
<div align="left"> <img width="1492" height="1000" alt="Image" src="09_assets/wbs.png" width="10%" height="10%"/> </div>

---

## 🛠️ 기술 스택

| 분류 | 기술                                                                                                                                                                                                                                                                                                   |
|------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 언어 | ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white)                                                                                                                                                                                                |
| 데이터 처리 | ![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=Pandas&logoColor=white) ![Numpy](https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=Numpy&logoColor=white)                                                                                             |
| 시각화 | ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white) ![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)                                                                              |
| 머신러닝 | ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white) ![XGBoost](https://img.shields.io/badge/XGBoost-EC5E0C?style=for-the-badge)                                                                                                                                                                            |
| 앱 프레임워크 | ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)                                                                                                                                                                                       |
| 파일 입출력 | ![JSON](https://img.shields.io/badge/json-%234479A1.svg?style=for-the-badge&logo=Json&logoColor=white) ![joblib](https://img.shields.io/badge/joblib-FFCA28?style=for-the-badge)|
| 버전 관리 | ![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)                                                                                                                                                                                                |

---
## 📁 폴더 구조
```
project/
├── 00_data/
│   ├── 00_raw/          # 원본 데이터 (수정 금지)
│   ├── 01_interim/      # 전처리 중간 산출물
│   └── 02_processed/    # 최종 학습/추론용 데이터
│
├── 01_notebooks/        # EDA · 실험용 노트북
│
├── 02_src/
│   ├── 00_common/       # 공통 유틸
│   ├── 01_data/         # 데이터 파이프라인 (전처리 · IO)
│   ├── 02_model/        # 모델 정의 · 학습 · 추론 · 저장
│   └── 03_front/        # Streamlit UI · 서비스 로직
│
├── 03_scripts/          # 파이프라인 실행 엔트리포인트
├── 04_configs/          # 모델 하이퍼파라미터 설정 (JSON)
├── 05_artifacts/        # 학습된 모델 · 전처리기 · 성능 지표
│
├── 09_assets/           # README 이미지, ERD, 파이프라인 다이어그램, 대시보드 스크린샷
│
├── app.py               # Streamlit 앱 진입점
├── requirements.txt
└── README.md
```



---

## 🗂️ 데이터

본 프로젝트는 두 가지 데이터를 단계적으로 활용합니다.

### 1차 데이터 — 한국미디어패널 (2023·2024)

이용자의 디지털 역량, 기술 인식, OTT 이용 행태 등 **설문 기반 개인 특성** 데이터입니다.

| 변수명 | 설명 | 유형 |
|--------|------|------|
| `use_infinite_data_plan` | 무제한 데이터 요금제 사용 여부 | 사용자 환경 |
| `pc_ability` / `internet_ability` | PC·인터넷 활용 능력 | 디지털 역량 |
| `mobile_skills` / `mobile_transaction_skills` | 모바일 활용·거래 능력 | 디지털 역량 |
| `privacy` / `newtech_perception` | 개인정보 보호 인식, 신기술 인식 | 기술 인식 |
| `ott_usage_weekday` / `ott_usage_weekend` | 평일·주말 OTT 이용 시간 | OTT 이용 행동 |
| `age_group` | 연령대 | 사용자 특성 |
| `is_churned` | OTT 서비스 이탈 여부 **(타겟 변수)** | 종속 변수 |

### 2차 데이터 — Netflix 사용자 행동 데이터 (약 21만 건+)

실제 서비스 이용 행동 로그 기반의 **행동 데이터**입니다. 최종 모델 학습 및 서비스에 활용됩니다.

| 파일 | 설명                   | 주요 컬럼                                                                                |
|------|----------------------|--------------------------------------------------------------------------------------|
| `netflix_users.csv` | 사용자 기본 정보            | `user_id`, `age`, `plan_tier`, `monthly_spend`, `is_active`                          |
| `netflix_watch_history.csv` | 시청 이력 로그             | `user_id`, `movie_id`, `watch_date`, `watch_duration_minutes`, `progress_percentage` |
| `user_data_to_analysis.csv` | streamlit 테스트 데이터 | 'is_active'를 제외한 사용자 기본 정보                                                           |

[//]: # (> ⚠️ 원본 데이터&#40;`00_data/00_raw/`&#41;는 수정 금지. 전처리 산출물은 `01_interim/`, 최종 학습 데이터는 `02_processed/`에 저장됩니다.)

---

## 🔍 EDA 주요 인사이트

- **이탈률 불균형**: 전체 데이터에서 이탈 고객(is_churned=1) 비율이 낮아 불균형 처리(class_weight, scale_pos_weight) 전략을 적용
- **최근 미시청 일수**: `days_since_last_watch`가 높을수록 이탈 확률이 유의미하게 증가 — Feature Importance 상위 변수로 확인
- **구독 유지 기간**: `subscription_tenure_days`가 짧은 초기 구독자일수록 이탈률이 높게 나타남
- **요금제 등급**: 저가 요금제(`plan_tier`) 사용자에서 이탈 경향이 상대적으로 강함
- **완주율 & 시청 시간**: `completion_rate`이 낮고 `total_watch_time`이 적은 사용자는 이탈 가능성이 높음 — 2차 데이터 XGBoost 기준 최상위 중요 피처로 확인
- **디지털 역량**: 1차 데이터 분석에서는 모바일 기기 활용 능력, 정보 평가 능력, 개인정보 인식 수준이 이탈과 연관됨

---
## ⚙️ 피처 엔지니어링

시청 이력 로그를 사용자 단위로 집계하여 아래 16개 피처를 생성합니다.

| 피처명 | 설명 |
|--------|------|
| `age` | 연령 |
| `plan_tier` | 요금제 등급 |
| `monthly_spend` | 월 지출액 |
| `age_group` | 연령대 (10단위 구간화) |
| `subscription_tenure_days` | 구독 유지 기간 (일) — 가입일로부터 직접 계산한 파생변수 |
| `watch_count` | 총 시청 횟수 |
| `unique_movies` | 시청한 고유 콘텐츠 수 (다양성) |
| `total_watch_time` | 총 시청 시간 (분) ⭐ |
| `avg_watch_time` | 평균 시청 시간 (분) |
| `watch_days` | 총 시청 일수 ⭐ |
| `recent_watch_count` | 최근 31일 시청 횟수 |
| `days_since_last_watch` | 마지막 시청 후 경과 일수 ⭐ |
| `avg_progress` | 평균 시청 진행률 (%) |
| `completion_rate` | 완주율 (90% 이상 시청 비율) |
| `download_ratio` | 다운로드 비율 |
| `avg_rating` | 평균 평점 |

> ⭐ XGBoost Feature Importance 분석 기준 상위 영향 변수

---

## 🤖 모델링

### 분석 전략

**1차 데이터(설문)** → 모델 구조 탐색 및 설문 변수의 예측 가능성 검증  
**2차 데이터(행동)** → 예측 성능 고도화 및 실제 서비스 적용 모델 구축

이탈 예측 특성상 **놓친 이탈자(FN)의 비용이 크기 때문에** Recall과 PR-AUC를 핵심 지표로 설정했습니다.

### 사용 모델

| 모델 | 스케일링 | 클래스 불균형 처리 | 비고 |
|------|----------|-------------------|------|
| Logistic Regression | ✅ StandardScaler | `class_weight=balanced` | 베이스라인 |
| Random Forest | ❌ | `class_weight=balanced_subsample` | 앙상블 |
| **XGBoost** | ❌ | `scale_pos_weight=2.645` | **최종 선택 모델** |

### 학습 설정 (공통)

- 교차 검증: `StratifiedKFold (k=5)` — 클래스 비율 유지하며 안정적 평가
- 데이터 분할: `train 80% / test 20%`, stratify 적용
- 평가 지표: `Precision`, `Recall`, `F1`, `ROC-AUC`, `PR-AUC`
- 최적 임계값 탐색: F1-score 기준 threshold 탐색 (0.1 ~ 0.9)

### 모델 학습 파이프라인
[Script 상세 실행 가이드](./03_scripts/README.md)
<div align="left"> <img width="1307" height="1397" alt="Image" src="09_assets/train_diagram.png" width="10%" height="10%" /> </div>

### 모델 추론 파이프라인
<div align="left"> <img width="1307" height="1397" alt="Image" src="09_assets/predict_diagram.png" width="10%" height="10%" /> </div>
---

### 📊 모델 성능 비교

#### 1차 데이터 결과 (설문 기반)

| 모델 | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|------|-----------|--------|----|---------|--------|
| Logistic Regression | 0.484 | 0.164 | 0.245 | 0.603 | 0.455 |
| Random Forest | 0.476 | 0.557 | 0.513 | 0.648 | 0.524 |
| XGBoost | 0.473 | 0.715 | 0.540 | 0.648 | 0.548 |

> 설문 기반 변수만으로는 실제 이탈 행동 예측에 한계가 있음을 확인 → 2차 데이터 분석으로 이어짐

#### 2차 데이터 결과 (행동 기반) — 최종 모델

| 모델 | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|------|-----------|--------|----|---------|--------|
| Logistic Regression | 0.796 | 0.879 | 0.835 | 0.925 | 0.877 |
| Random Forest | 0.867 | 0.883 | 0.875 | 0.912 | 0.939 |
| **XGBoost** | **0.893** | **0.880** | **0.887** | **0.922** | **0.945** |

> XGBoost가 Precision·Recall·F1 모두 최고 수준. PR-AUC 0.945로 이탈 고객 탐지 능력 검증 완료

---

### 하이퍼파라미터 설정

#### Logistic Regression

| 파라미터 | 값 | 설명 |
|----------|----|------|
| `C` | 0.01 | 규제 강도 (작을수록 강한 규제) |
| `penalty` | l2 | Ridge 규제 적용 |
| `solver` | liblinear | 이진 분류에 적합한 알고리즘 |
| `class_weight` | balanced | 클래스 불균형 보정 |

#### Random Forest

| 파라미터 | 값 | 설명 |
|----------|----|------|
| `n_estimators` | 500 | 생성할 결정 트리 수 |
| `max_depth` | 5 | 트리 최대 깊이 |
| `min_samples_split` | 10 | 노드 분할 최소 샘플 수 |
| `max_features` | None | 분할 시 전체 변수 사용 |
| `class_weight` | balanced | 클래스 불균형 보정 |

#### XGBoost

| 파라미터 | 값 | 설명 |
|----------|----|------|
| `n_estimators` | 500 | 생성할 트리 수 |
| `max_depth` | 3 | 트리 최대 깊이 |
| `learning_rate` | 0.03 | 학습률 (낮을수록 꼼꼼하게 학습) |
| `subsample` | 0.7 | 트리 학습 시 데이터 샘플 비율 |
| `colsample_bytree` | 0.7 | 트리 학습 시 피처 사용 비율 |
| `min_child_weight` | 3 | 리프 노드 최소 가중치 합 |
| `gamma` | 0 | 노드 분할 최소 손실 감소값 |
| `scale_pos_weight` | 2.645 | 이탈/비이탈 클래스 불균형 보정 가중치 |

---

## 🖥️ Streamlit 앱

분석 결과 가장 높은 성능을 보인 **2차 데이터 기반 XGBoost 모델**을 적용한 웹 대시보드입니다.

> 📸 **스크린샷 / GIF**  


<div align="left"> <img width="1307" height="1500" alt="Image" src="09_assets/dashboard.png" width="10%" height="10%" /> </div>


### 주요 기능

- **홈 화면**: CSV 파일 업로드 및 전체 고객 현황 확인 (고객 수, 이탈 위험 비율, 이용 패턴)
- **대시보드**: 개별 고객 이탈 확률 예측 + 위험군 자동 분류 (위험군 / 잠재 위험군 / 유지군)
- **시각화**: Feature Importance, PR-Curve 차트 제공
- 배치 CSV 업로드 방식으로 대규모 고객 데이터 일괄 처리 지원

---

## ⚠️ 한계점 및 향후 개선 방향

### 현재 한계점

- **단일 플랫폼 데이터**: Netflix 데이터만 활용하여 Disney+, 티빙, 웨이브 등 전체 OTT 시장으로 일반화하는 데 제약이 있음
- **행동 변수 다양성 부족**: 콘텐츠 장르 선호도, 추천 시스템 반응, UI 이용 패턴 등 추가 행동 데이터 미반영
- **정적 배치 방식**: 실시간 스트리밍 없이 CSV 업로드 방식으로만 추론 가능
- **모델 범위**: 머신러닝 3종 비교에 그치며, 딥러닝·하이브리드 앙상블 등 추가 실험 미수행
- **설명 가능성**: 피처 중요도 수준의 해석만 제공, 개인별 예측 근거 설명 미구현

### 향후 개선 방향

- [ ] **다양한 OTT 플랫폼 데이터 통합** 분석으로 일반화 가능성 확대
- [ ] **SHAP 기반 개인별 예측 설명** 추가 (왜 이 고객이 이탈 위험군인가?)
- [ ] **딥러닝 모델 실험** (TabNet, LSTM 기반 시계열 행동 패턴 학습)
- [ ] **실시간 추론 API** 구축 (FastAPI + 배치 스케줄링)
- [ ] **모델 모니터링 & 자동 재학습** 파이프라인 (MLflow 연동)
- [ ] **콘텐츠 장르 선호도, 추천 반응 데이터** 등 행동 변수 확장

---

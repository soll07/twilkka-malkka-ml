# models/ 폴더

EDA에서 학습한 모델 pkl 파일을 이 폴더에 배치해 주세요.

## 필요한 파일

| 파일명 | 설명 |
|--------|------|
| `xgboost.pkl` | XGBoost 학습 모델 |
| `random_forest.pkl` | Random Forest 학습 모델 |
| `logistic_regression.pkl` | Logistic Regression 학습 모델 |
| `preprocessor.pkl` | (선택) sklearn Pipeline/ColumnTransformer 전처리기 |

## 저장 방법 (EDA 노트북에서)

```python
import joblib

# 모델 저장
joblib.dump(xgb_model, "models/xgboost.pkl")
joblib.dump(rf_model,  "models/random_forest.pkl")
joblib.dump(lr_model,  "models/logistic_regression.pkl")

# 전처리 파이프라인도 함께 저장하면 자동 적용됩니다
joblib.dump(preprocessor, "models/preprocessor.pkl")
```

## 파일이 없는 경우
- `models/` 폴더에 pkl 파일이 없으면 **내장 규칙 기반 예측(fallback)**으로 동작합니다.
- 샘플 데이터로 보기 버튼은 항상 동작합니다.

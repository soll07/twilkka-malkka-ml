"""
predictor.py
============
EDA에서 학습한 3개 모델(Logistic Regression, Random Forest, XGBoost)을 로드하고
업로드된 CSV/XLSX 데이터에 대해 이탈 확률을 예측한 뒤,
대시보드에 필요한 모든 집계 데이터를 반환합니다.

모델 파일 위치: dasol_model/models/
  - logistic_regression.pkl
  - random_forest.pkl
  - xgboost.pkl
  - preprocessor.pkl  (sklearn Pipeline/ColumnTransformer, 없으면 내부 전처리 사용)
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── 경로 설정 ──────────────────────────────────────────────────────────────
_BASE_DIR = Path(__file__).resolve().parents[3]   # dasol_model/
MODELS_DIR = _BASE_DIR / "models"

MODEL_FILES = {
    "XGBoost": "xgboost.pkl",
    "Random Forest": "random_forest.pkl",
    "Logistic Regression": "logistic_regression.pkl",
}
PREPROCESSOR_FILE = "preprocessor.pkl"

# ── 위험도 분류 임계값 ──────────────────────────────────────────────────────
RISK_BINS   = [0.0, 0.30, 0.50, 0.70, 1.01]
RISK_LABELS = ["안전", "낮은 위험", "중간 위험", "높은 위험"]
RISK_ORDER  = ["높은 위험", "중간 위험", "낮은 위험", "안전"]


# ── 모델 로드 ──────────────────────────────────────────────────────────────

def load_model(model_name: str) -> Any:
    """pkl 파일에서 학습된 모델을 불러옵니다."""
    filename = MODEL_FILES.get(model_name)
    if filename is None:
        raise ValueError(f"알 수 없는 모델: {model_name}")
    path = MODELS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {path}")
    return joblib.load(path)


def load_preprocessor() -> Any | None:
    """전처리 파이프라인을 불러옵니다. 없으면 None 반환."""
    path = MODELS_DIR / PREPROCESSOR_FILE
    if path.exists():
        return joblib.load(path)
    return None


# ── 내부 전처리 (preprocessor.pkl 없을 때 fallback) ───────────────────────

def _auto_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    범주형 → Label Encoding, 결측치 → 중앙값/최빈값으로 간단 처리.
    EDA에서 사용한 전처리 파이프라인(preprocessor.pkl)이 있으면 이 함수는 호출되지 않습니다.
    """
    df = df.copy()

    # 타겟 컬럼 후보 제거 (있을 경우)
    for col in ["Churn", "churn", "이탈", "label", "target"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    for col in df.columns:
        if df[col].dtype == "object" or str(df[col].dtype) == "category":
            df[col] = df[col].astype("category").cat.codes
            df[col] = df[col].replace(-1, np.nan)

    for col in df.select_dtypes(include="number").columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    return df


# ── 핵심 예측 함수 ─────────────────────────────────────────────────────────

def predict_churn(
    df: pd.DataFrame,
    model_name: str = "XGBoost",
) -> pd.DataFrame:
    """
    업로드된 DataFrame에 대해 이탈 확률을 예측하고
    churn_prob, risk_label 컬럼을 추가한 DataFrame을 반환합니다.
    """
    model = load_model(model_name)
    preprocessor = load_preprocessor()

    raw_df = df.copy()

    # 타겟 컬럼 분리 (있을 경우)
    target_col = None
    for col in ["Churn", "churn", "이탈", "label", "target"]:
        if col in raw_df.columns:
            target_col = col
            break

    if preprocessor is not None:
        X = preprocessor.transform(raw_df.drop(columns=[target_col]) if target_col else raw_df)
    else:
        X = _auto_preprocess(raw_df).values

    probs = model.predict_proba(X)[:, 1]

    result = raw_df.copy()
    result["churn_prob"] = probs
    result["risk_label"] = pd.cut(
        probs,
        bins=RISK_BINS,
        labels=RISK_LABELS,
        right=False,
    ).astype(str)

    return result


# ── 대시보드용 집계 함수들 ─────────────────────────────────────────────────

def build_kpi(result_df: pd.DataFrame) -> list[dict]:
    """KPI 카드 3개 데이터를 반환합니다."""
    total = len(result_df)
    high_risk = (result_df["risk_label"] == "높은 위험").sum()
    retention = (total - high_risk) / total * 100 if total else 0

    return [
        {
            "title": "총 사용자",
            "value": f"{total:,}",
            "delta": f"{total:,}명",
            "delta_type": "positive",
            "icon": "👥",
        },
        {
            "title": "예상 이탈 사용자",
            "value": f"{high_risk:,}",
            "delta": f"{high_risk / total * 100:.1f}%" if total else "0%",
            "delta_type": "negative",
            "icon": "🚨",
        },
        {
            "title": "구독 유지율",
            "value": f"{retention:.1f}%",
            "delta": f"{retention:.1f}%",
            "delta_type": "positive" if retention >= 85 else "negative",
            "icon": "📊",
        },
    ]


def build_risk_segments(result_df: pd.DataFrame) -> dict:
    """위험도별 사용자 수 분포를 반환합니다."""
    counts = result_df["risk_label"].value_counts()
    values = [int(counts.get(label, 0)) for label in RISK_ORDER]
    return {"labels": RISK_ORDER, "values": values}


def build_churn_drivers(model_name: str, feature_names: list[str]) -> list[dict]:
    """
    모델의 feature importance 상위 4개를 이탈 예측 주요 요인으로 반환합니다.
    feature importance를 지원하지 않는 모델은 기본값 반환.
    """
    try:
        model = load_model(model_name)

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
        else:
            return _default_churn_drivers()

        total = importances.sum()
        if total == 0:
            return _default_churn_drivers()

        idx_sorted = np.argsort(importances)[::-1][:4]
        result = []
        for i in idx_sorted:
            name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
            pct  = int(round(importances[i] / total * 100))
            result.append({"label": name, "value": pct})
        return result

    except Exception:
        return _default_churn_drivers()


def _default_churn_drivers() -> list[dict]:
    return [
        {"label": "비활성 일수",       "value": 32},
        {"label": "주간 시청 시간 감소", "value": 24},
        {"label": "로그인 빈도 감소",   "value": 18},
        {"label": "콘텐츠 다양성 부족", "value": 14},
    ]


def build_ott_usage(result_df: pd.DataFrame) -> list[dict]:
    """
    시청 빈도 관련 컬럼이 있으면 실제 데이터 기반, 없으면 기본값 반환.
    컬럼 후보: ViewingFrequency, viewing_frequency, 시청빈도
    """
    freq_col = _find_col(result_df, ["ViewingFrequency", "viewing_frequency", "시청빈도", "UsageFrequency"])
    if freq_col is None:
        return [
            {"label": "매일",     "value": 42},
            {"label": "주 3~4회", "value": 28},
            {"label": "주 1~2회", "value": 18},
            {"label": "월 1~3회", "value": 12},
        ]

    counts = result_df[freq_col].value_counts(normalize=True) * 100
    return [
        {"label": str(label), "value": int(round(pct))}
        for label, pct in counts.head(4).items()
    ]


def build_genres(result_df: pd.DataFrame) -> list[dict]:
    """
    장르 관련 컬럼이 있으면 실제 데이터 기반, 없으면 기본값 반환.
    컬럼 후보: Genre, genre, 장르, FavoriteGenre
    """
    genre_col = _find_col(result_df, ["Genre", "genre", "장르", "FavoriteGenre", "favorite_genre"])
    if genre_col is None:
        return [
            {"label": "드라마", "value": 38},
            {"label": "영화",   "value": 27},
            {"label": "예능",   "value": 19},
            {"label": "다큐",   "value": 10},
            {"label": "애니",   "value":  6},
        ]

    counts = result_df[genre_col].value_counts(normalize=True) * 100
    return [
        {"label": str(label), "value": int(round(pct))}
        for label, pct in counts.head(5).items()
    ]


def build_monthly_trend(result_df: pd.DataFrame) -> dict:
    """
    월별 이탈 예측 확률 추세 (실제 날짜 컬럼이 있으면 기반, 없으면 예측 확률 분포 기반).
    """
    date_col = _find_col(result_df, ["date", "Date", "month", "Month", "날짜", "가입일", "JoinDate"])
    months = ["1월","2월","3월","4월","5월","6월","7월","8월","9월","10월","11월","12월"]

    if date_col and "churn_prob" in result_df.columns:
        try:
            tmp = result_df.copy()
            tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
            tmp = tmp.dropna(subset=[date_col])
            tmp["month"] = tmp[date_col].dt.month
            monthly = tmp.groupby("month")["churn_prob"].mean() * 100
            predicted = [round(float(monthly.get(m, 65)), 1) for m in range(1, 13)]
            high_risk_rate = tmp.groupby("month").apply(
                lambda x: (x["risk_label"] == "높은 위험").mean() * 100
            )
            actual = [round(float(high_risk_rate.get(m, 60)), 1) for m in range(1, 13)]
            return {"months": months, "이탈률": predicted, "시청감소율": actual}
        except Exception:
            pass

    # fallback: 예측 확률 기반 히스토그램 분포로 트렌드 추정
    if "churn_prob" in result_df.columns:
        avg_prob = float(result_df["churn_prob"].mean()) * 100
        trend = [round(avg_prob * (0.75 + 0.04 * i), 1) for i in range(12)]
        trend2 = [round(avg_prob * (0.72 + 0.04 * i), 1) for i in range(12)]
        return {"months": months, "이탈률": trend, "시청감소율": trend2}

    return {
        "months": months,
        "이탈률":    [55,60,63,67,70,71,75,76,73,71,69,67],
        "시청감소율": [52,58,61,66,71,69,74,77,74,72,70,68],
    }


def build_high_risk_users(result_df: pd.DataFrame, top_n: int = 5) -> list[dict]:
    """이탈 확률 상위 N명의 사용자 정보를 반환합니다."""
    if "churn_prob" not in result_df.columns:
        return []

    top = result_df.nlargest(top_n, "churn_prob").copy()

    name_col   = _find_col(top, ["CustomerID", "customer_id", "Name", "name", "사용자", "고객ID", "UserID"])
    email_col  = _find_col(top, ["Email", "email", "이메일"])
    active_col = _find_col(top, ["LastLogin", "last_login", "마지막로그인", "LastActive", "last_active"])
    days_col   = _find_col(top, ["InactiveDays", "inactive_days", "비활성일수", "DaysSinceLastLogin"])
    genre_col  = _find_col(top, ["Genre", "genre", "장르", "FavoriteGenre"])

    users = []
    for i, (_, row) in enumerate(top.iterrows()):
        users.append({
            "name":          str(row[name_col])  if name_col   else f"고객 #{i+1}",
            "email":         str(row[email_col]) if email_col  else "정보 없음",
            "last_active":   str(row[active_col])if active_col else "-",
            "inactive_days": f"{int(row[days_col])}일" if days_col and pd.notna(row[days_col]) else "-",
            "genre":         str(row[genre_col]) if genre_col  else "-",
            "risk":          f"{row['churn_prob']*100:.0f}%",
            "risk_label":    str(row["risk_label"]),
        })
    return users


# ── 유틸 ───────────────────────────────────────────────────────────────────

def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """후보 컬럼명 중 DataFrame에 존재하는 첫 번째를 반환합니다."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def get_feature_names(df: pd.DataFrame) -> list[str]:
    """전처리 후 사용되는 피처 이름 목록을 반환합니다."""
    preprocessor = load_preprocessor()
    if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
        try:
            return list(preprocessor.get_feature_names_out())
        except Exception:
            pass
    drop_cols = ["Churn", "churn", "이탈", "label", "target", "churn_prob", "risk_label"]
    return [c for c in df.columns if c not in drop_cols]


def models_available() -> list[str]:
    """실제로 파일이 존재하는 모델 목록을 반환합니다."""
    available = []
    for name, filename in MODEL_FILES.items():
        if (MODELS_DIR / filename).exists():
            available.append(name)
    return available

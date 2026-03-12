"""
data.py  - 샘플 DataFrame 생성 + 하드코딩 fallback 상수
"""
from __future__ import annotations
import datetime, random
import numpy as np
import pandas as pd

_now = datetime.datetime.now()
_ampm = "오전" if _now.hour < 12 else "오후"
_hour = _now.hour if _now.hour <= 12 else _now.hour - 12
NAVBAR_UPDATE = f"{_now.year}년 {_now.month}월 {_now.day}일 · {_ampm} {_hour}:{_now.minute:02d}"


def make_sample_dataframe(n: int = 2050, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    random.seed(seed)
    genres      = ["드라마","영화","예능","다큐","애니"]
    genre_probs = [0.38,0.27,0.19,0.10,0.06]
    freq_labels = ["매일","주 3~4회","주 1~2회","월 1~3회"]
    freq_probs  = [0.42,0.28,0.18,0.12]

    inactive_days      = rng.integers(0, 60, size=n)
    weekly_watch_hours = rng.exponential(scale=8, size=n).round(1)
    login_freq         = rng.integers(1, 30, size=n)
    content_diversity  = rng.integers(1, 10, size=n)

    churn_score = (
        0.40 * (inactive_days / 60)
        + 0.25 * (1 - weekly_watch_hours / (weekly_watch_hours.max() + 1e-9))
        + 0.20 * (1 - login_freq / 30)
        + 0.15 * (1 - content_diversity / 10)
        + rng.uniform(-0.05, 0.05, size=n)
    )
    churn_score  = np.clip(churn_score, 0.01, 0.99)
    churn_label  = (churn_score > 0.5).astype(int)
    customer_ids = [f"CUST{str(i+1).zfill(5)}" for i in range(n)]
    base_date    = datetime.date(2026, 3, 1)
    last_logins  = [str(base_date - datetime.timedelta(days=int(d))) for d in inactive_days]

    return pd.DataFrame({
        "CustomerID":       customer_ids,
        "InactiveDays":     inactive_days,
        "WeeklyWatchHours": weekly_watch_hours,
        "LoginFrequency":   login_freq,
        "ContentDiversity": content_diversity,
        "FavoriteGenre":    rng.choice(genres, size=n, p=genre_probs),
        "ViewingFrequency": rng.choice(freq_labels, size=n, p=freq_probs),
        "LastActive":       last_logins,
        "Churn":            churn_label,
    })


# ── fallback 상수 ──────────────────────────────────────────────────────────
KPI_DATA = [
    {"title": "총 사용자",       "value": "2,050", "delta": "+5.2%",  "delta_type": "positive", "icon": "👥"},
    {"title": "예상 이탈 사용자", "value": "247",   "delta": "+12.3%", "delta_type": "negative", "icon": "🚨"},
    {"title": "구독 유지율",     "value": "87.9%", "delta": "-1.1%",  "delta_type": "negative", "icon": "📊"},
]
MONTHLY_TREND = {
    "months":   ["1월","2월","3월","4월","5월","6월","7월","8월","9월","10월","11월","12월"],
    "이탈률":    [55,60,63,67,70,71,75,76,73,71,69,67],
    "시청감소율": [52,58,61,66,71,69,74,77,74,72,70,68],
}
RISK_SEGMENTS = {"labels": ["높은 위험","중간 위험","낮은 위험","안전"], "values": [247,589,823,391]}
OTT_USAGE     = [{"label":"매일","value":42},{"label":"주 3~4회","value":28},{"label":"주 1~2회","value":18},{"label":"월 1~3회","value":12}]
GENRES        = [{"label":"드라마","value":38},{"label":"영화","value":27},{"label":"예능","value":19},{"label":"다큐","value":10},{"label":"애니","value":6}]
CHURN_DRIVERS = [{"label":"비활성 일수","value":32},{"label":"주간 시청 시간 감소","value":24},{"label":"로그인 빈도 감소","value":18},{"label":"콘텐츠 다양성 부족","value":14}]
HIGH_RISK_USERS = [
    {"name":"Sarah Johnson","email":"sarah.j@email.com","last_active":"2026.02.17","inactive_days":"18일","genre":"드라마","risk":"87%","risk_label":"높은 위험"},
    {"name":"David Miller", "email":"david.m@email.com","last_active":"2026.02.15","inactive_days":"20일","genre":"영화",  "risk":"84%","risk_label":"높은 위험"},
    {"name":"Emma Wilson",  "email":"emma.w@email.com", "last_active":"2026.02.18","inactive_days":"16일","genre":"예능",  "risk":"81%","risk_label":"높은 위험"},
]

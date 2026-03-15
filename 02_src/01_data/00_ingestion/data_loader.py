import pandas as pd
from pathlib import Path

class DataLoader:
    RAW_COLS = ["user_id", "age", "plan_tier", "subscription_start_date", "monthly_spend", "is_active"]
    FEATURE_COLS = ["age", "plan_tier", "monthly_spend", "age_group", "subscription_tenure_days",
        "watch_count", "unique_movies", "total_watch_time", "avg_watch_time", "watch_days",
        "recent_watch_count", "days_since_last_watch", "avg_progress",
        "completion_rate", "download_ratio", "avg_rating"]
    TRAIN_COLS = FEATURE_COLS + ["is_active"]
    PREDICT_COLS = ["user_id", "age", "plan_tier", "subscription_start_date", "monthly_spend"]

    SCHEMAS = {
        'raw':RAW_COLS,
        'feature':FEATURE_COLS,
        'train': TRAIN_COLS,
        'predict':PREDICT_COLS,
    }

    @classmethod
    def load_csv(cls, file_path: Path) -> pd.DataFrame:
        """
        csv 파일 로드 및 완전 중복 제거
        """
        if not file_path.exists():
            raise FileNotFoundError(f"❌ 경로에 파일이 없습니다: {file_path}")
        df = pd.read_csv(file_path)
        return df.drop_duplicates(keep='last')

    @classmethod
    def validate_df(cls, df: pd.DataFrame, mode: str) -> tuple[bool, list[str]]:
        """
        필수 컬럼 존재 여부 확인 로직
        """
        target_schema = set(cls.SCHEMAS[mode])
        current_columns = set(df.columns)
        missing_columns = list(target_schema - current_columns)

        return len(missing_columns) == 0, missing_columns

    @classmethod
    def reorder_columns(cls, df: pd.DataFrame, mode: str) -> pd.DataFrame:
        """
        스키마에 정의된 순서로 컬럼 순서 재배치
        """
        target_schema = cls.SCHEMAS[mode]
        return  df[target_schema]

import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
INTERIM_DIR = ROOT / "00_data" / "01_interim"
WATCH_FEATURES_PATH = INTERIM_DIR / "watch_features.csv"

def process_uploaded_file(uploaded_file):
    user_df = pd.read_csv(uploaded_file)
    return create_inference_data(user_df)

def create_inference_data(user_df):
    raw_df = user_df.copy()

    # 유저별 시청 기록 집계 파일 생성
    if not WATCH_FEATURES_PATH.exists():
        generate_and_save_watch_features()
    # 이미 있는 경우 로드
    watch_features = pd.read_csv(WATCH_FEATURES_PATH)

    # watch_features에 저장된 기준 일자
    if 'last_watch_date_ref' in watch_features.columns:
        ref_date_str = watch_features['last_watch_date_ref'].iloc[0]
        reference_date = pd.to_datetime(ref_date_str)
    else:
        reference_date = pd.Timestamp.now()

    # 사용자 데이터 전처리
    processed_user_df = (
        raw_df.pipe(select_columns)
          .pipe(clean_age)
          .pipe(add_age_group)
          .pipe(fill_monthly_spend_nan)
          .pipe(process_dates, reference_date)
    )


    watch_cols = [c for c in watch_features.columns if c != 'last_watch_date_ref'] # 집계용 변수 제외
    inference_data = (
        processed_user_df
        .merge(watch_features[watch_cols], on="user_id", how="left")
        .fillna(0)
    )
    
    # 시청 기록이 전혀 없는 경우 999일로 처리
    if "days_since_last_watch" in inference_data.columns:
        inference_data["days_since_last_watch"] = inference_data["days_since_last_watch"].replace(0, 999)

    return inference_data

def select_columns(df):
    user_columns = ['user_id', 'age', 'plan_tier',
                    'subscription_start_date', 'is_active', 'monthly_spend']
    #존재하는 컬럼만 선택
    existing_cols = [c for c in user_columns if c in df.columns]
    return df[existing_cols].copy()

def clean_age(df):
    if 'age' in df.columns:
        df.loc[(df["age"] < 0) | (df["age"] > 100), "age"] = np.nan
        df['age'] = df['age'].fillna(df['age'].median())
    return df

def add_age_group(df):
    if 'age' in df.columns:
        df['age_group'] = (df['age'] // 10).fillna(0).astype(int)
    return df

def fill_monthly_spend_nan(df):
    if 'monthly_spend' in df.columns:
        df['monthly_spend'] = df['monthly_spend'].fillna(0)
    return df

def process_dates(df, reference_date):

    if "subscription_start_date" in df.columns:
        df["subscription_start_date"] = pd.to_datetime(df["subscription_start_date"], errors='coerce')
        # NaT인 경우 기준일로 채워 구독 기간이 0이 되게 함
        df["subscription_start_date"] = df["subscription_start_date"].fillna(reference_date)
        

        tenure = (reference_date - df["subscription_start_date"]) # 구독 시작부터 며칠이 지났는가? (구독 기간)
        df["subscription_tenure_days"] = tenure.dt.days
        return df.drop(columns=['subscription_start_date'])
    return df

def generate_and_save_watch_features():

    raw_path = ROOT / "00_data" / "00_raw" / "netflix_watch_history.csv" #수정 예정
    if not raw_path.exists():
        print(f"Error: {raw_path} not found.")
        return

    print("Pre-calculating watch features...")
    history_df = pd.read_csv(raw_path)
    
    # 전처리
    history_df['watch_date'] = pd.to_datetime(history_df['watch_date'], errors='coerce')
    history_df["watch_duration_minutes"] = history_df["watch_duration_minutes"].fillna(0)
    history_df["progress_percentage"] = history_df["progress_percentage"].fillna(0)
    
    last_date = history_df['watch_date'].max()

    # 집계
    watch_count = history_df.groupby('user_id').size().rename('watch_count')
    unique_movies = history_df.groupby("user_id")["movie_id"].nunique().rename("unique_movies")
    watch_time = history_df.groupby("user_id")["watch_duration_minutes"].agg(
        total_watch_time="sum",
        avg_watch_time="mean"
    )
    watch_days = history_df.groupby("user_id")["watch_date"].nunique().rename("watch_days")
    
    recent = history_df[history_df['watch_date'] >= last_date - pd.Timedelta(days=31)]
    recent_watch_count = recent.groupby('user_id').size().rename('recent_watch_count')
    
    last_watch = history_df.groupby("user_id")["watch_date"].max()
    days_since_last_watch = (last_date - last_watch).dt.days.rename("days_since_last_watch")
    
    avg_progress = history_df.groupby("user_id")["progress_percentage"].mean().rename('avg_progress')
    history_df['completed'] = history_df['progress_percentage'] >= 90
    completion_rate = history_df.groupby("user_id")["completed"].mean().rename('completion_rate')
    
    download_ratio = history_df.groupby('user_id')['is_download'].mean().rename('download_ratio')
    avg_rating = history_df.groupby("user_id")["user_rating"].mean().rename('avg_rating')
    
    # 결합
    watch_features = pd.concat([
        watch_count, unique_movies, watch_time, watch_days,
        recent_watch_count, days_since_last_watch, avg_progress,
        completion_rate, download_ratio, avg_rating
    ], axis=1).reset_index()
    
    # 기준 날짜 저장 (나중에 추론 시 tenure 계산용)
    watch_features['last_watch_date_ref'] = last_date
    
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    watch_features.to_csv(WATCH_FEATURES_PATH, index=False)
    print(f"Saved pre-calculated features to {WATCH_FEATURES_PATH}")

if __name__ == "__main__":
    USER_PATH = ROOT / "00_data" / "01_interim" / "user_data_for_input.csv"
    test_train_data = process_uploaded_file(USER_PATH)

    SAVE_PATH = ROOT / "00_data" / "01_interim" / "inference_test_data.csv"
    test_train_data.to_csv(SAVE_PATH, index=False)

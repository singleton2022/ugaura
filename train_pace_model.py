import sqlite3
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
import data_loader

DB_PATH = "C:/Ugaura/sqlite/jra_race.db"
MODEL_FILENAME = "lgbm_pace_model.txt"

def train_pace_model():
    print("1. data_loader から特徴量マトリクスをロード中...")
    # データベースから全馬の特徴量データをロード
    df_raw = data_loader.load_feature_matrix(DB_PATH)
    
    # レース単位に集約するカラムの定義
    race_cols = [
        'race_date', 'distance', 'course_code', 'track_code', 
        'turf_condition_code', 'dirt_condition_code', 
        'pace_score_top3', 'leader_margin', 
        'straight_length', 'is_steep_slope', 'is_local',
        'escape_horse_count_in_race', 'lead_horse_count_in_race',
        'front_active_horse_count_in_race', 'front_active_horse_ratio',
        'weather_code', 'grade_code', 'race_type_code', 'weight_type_code',
        'cond_code_youngest', 'race_horse_count'
    ]
    actual_race_cols = [col for col in race_cols if col in df_raw.columns]
    
    # レースID単位で1行に集約
    df_race = df_raw.groupby('race_id')[actual_race_cols].first().reset_index()
    if 'race_horse_count' in df_race.columns:
        df_race = df_race.rename(columns={'race_horse_count': 'started_count'})
    print(f"集約完了: レース数={len(df_race)}件")
    
    # 2. 目的変数（ラップタイム差）を race_lap_time テーブルから計算してマージ
    print("2. race_lap_time からラップデータをロードして目的変数を算出中...")
    query_lap = """
    SELECT 
        year || month_day || course_code || times || day || race_number AS race_id,
        CAST(lap_number AS INTEGER) AS lap_number,
        CAST(lap_time AS INTEGER) AS lap_time_val
    FROM race_lap_time
    WHERE lap_time IS NOT NULL AND lap_time != ''
    """
    with sqlite3.connect(DB_PATH) as conn:
        df_lap = pd.read_sql_query(query_lap, conn)
        
    df_lap['lap_time'] = df_lap['lap_time_val'] / 10.0
    df_lap['max_lap'] = df_lap.groupby('race_id')['lap_number'].transform('max')
    
    # 3ハロン以上のデータが存在するレースのみ対象
    valid_lap_races = df_lap[df_lap['max_lap'] >= 3]['race_id'].unique()
    df_lap = df_lap[df_lap['race_id'].isin(valid_lap_races)].copy()
    
    front_3f = df_lap[df_lap['lap_number'] <= 3].groupby('race_id')['lap_time'].sum().rename('front_3f')
    back_3f = df_lap[df_lap['lap_number'] > df_lap['max_lap'] - 3].groupby('race_id')['lap_time'].sum().rename('back_3f')
    
    df = df_race.merge(front_3f, on='race_id', how='inner')
    df = df.merge(back_3f, on='race_id', how='inner')
    
    # 目的変数の算出（前3F - 後3F）
    df['lap_diff'] = df['front_3f'] - df['back_3f']
    
    # タイムの異常値除外
    df = df[(df['front_3f'] > 10.0) & (df['front_3f'] < 100.0)].copy()
    df = df[(df['back_3f'] > 10.0) & (df['back_3f'] < 100.0)].copy()
    
    # カテゴリ変数の型変換
    cat_features = [
        'course_code', 'track_code', 'turf_condition_code', 'dirt_condition_code',
        'weather_code', 'grade_code', 'race_type_code', 'weight_type_code', 'cond_code_youngest'
    ]
    for col in cat_features:
        df[col] = df[col].astype(str).fillna('-1').str.strip().astype('category')
        
    # 説明変数の定義（物理形状特性や逃げ頭数などの高度特徴量を追加）
    features = [
        'distance', 'course_code', 'track_code', 
        'turf_condition_code', 'dirt_condition_code', 
        'started_count', 'pace_score_top3', 'leader_margin',
        'straight_length', 'is_steep_slope', 'is_local',
        'escape_horse_count_in_race', 'lead_horse_count_in_race',
        'front_active_horse_count_in_race', 'front_active_horse_ratio',
        'weather_code', 'grade_code', 'race_type_code', 'weight_type_code', 'cond_code_youngest'
    ]
    target = 'lap_diff'
    
    # 時間によるデータ分割
    train_df = df[df['race_date'] <= '2023-12-31'].copy()
    valid_df = df[(df['race_date'] >= '2024-01-01') & (df['race_date'] <= '2024-12-31')].copy()
    test_df = df[df['race_date'] >= '2025-01-05'].copy()
    
    print(f"データ分割数: 訓練={len(train_df)}件, 検証={len(valid_df)}件, テスト={len(test_df)}件")
    
    if len(train_df) == 0 or len(valid_df) == 0:
        print("エラー: 訓練または検証データが足りません。")
        return
        
    lgb_train = lgb.Dataset(train_df[features], label=train_df[target], categorical_feature=cat_features)
    lgb_valid = lgb.Dataset(valid_df[features], label=valid_df[target], reference=lgb_train, categorical_feature=cat_features)
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.05,
        'max_depth': 5,
        'num_leaves': 31,
        'min_data_in_leaf': 50,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'random_state': 42,
        'verbose': -1
    }
    
    callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=True)]
    
    print("3. ペース予測回帰モデル（説明変数強化版）の学習開始...")
    booster = lgb.train(
        params, lgb_train, num_boost_round=5000,
        valid_sets=[lgb_train, lgb_valid], valid_names=['train', 'valid'], callbacks=callbacks
    )
    
    # 保存
    booster.save_model(MODEL_FILENAME, num_iteration=booster.best_iteration)
    print(f"モデルを保存しました: {MODEL_FILENAME}")
    
    # 評価
    y_pred_val = booster.predict(valid_df[features])
    rmse_val = np.sqrt(mean_squared_error(valid_df[target], y_pred_val))
    r2_val = r2_score(valid_df[target], y_pred_val)
    print(f"\n検証データ評価: RMSE = {rmse_val:.4f}, R2 = {r2_val:.4f}")
    
    if len(test_df) > 0:
        y_pred_test = booster.predict(test_df[features])
        rmse_test = np.sqrt(mean_squared_error(test_df[target], y_pred_test))
        r2_test = r2_score(test_df[target], y_pred_test)
        print(f"テストデータ評価: RMSE = {rmse_test:.4f}, R2 = {r2_test:.4f}")

    importance_df = pd.DataFrame({
        'feature': features,
        'importance': booster.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    print("\n--- Feature Importance (Pace Model - Enhanced) ---")
    print(importance_df.to_string(index=False))

if __name__ == "__main__":
    train_pace_model()

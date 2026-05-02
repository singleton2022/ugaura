import sqlite3
import pandas as pd
import numpy as np
import lightgbm as lgb
from data_loader import load_feature_matrix

# --- 設定 ---
TEST_START_DATE = '2025-01-05'  # （既存の開始日付の変数名に合わせてください）
TEST_END_DATE = '2026-04-26'    # ★新規追加: 集計の最終日付

def prepare_features(df):
    categorical_features = [
        'course_code', 'track_code', 'turf_condition_code', 'dirt_condition_code',
        'bracket_number', 'horse_number', 'sex_code', 'distance'
    ]
    for col in categorical_features:
        if df[col].dtype.name == 'category':
            if -1 not in df[col].cat.categories:
                df[col] = df[col].cat.add_categories([-1])
            df[col] = df[col].fillna(-1)
        else:
            df[col] = df[col].fillna(-1).astype('category')
    return df, categorical_features

def run_lambdarank_backtest(db_path='C:/sqlite/jra_race.db'):
    print("--- 芝/ダート分離 LambdaRank バックテスト開始 ---")
    df = load_feature_matrix(db_path)
    # test_df = df[df['race_date'] >= '2025-01-01'].copy()
    # 開始日付と終了日付の両方の条件を満たすデータを抽出
    test_df = df[(df['race_date'] >= TEST_START_DATE) & (df['race_date'] <= TEST_END_DATE)].copy()    

    numeric_features = [
        'pace_score_top3',
        'prev_ucv_score', 'max_ucv_score_5', 'avg_ucv_score_5', 
        'avg_ucv_3f_5', 'max_ucv_3f_5', 'ucv_3f_gap_avg_5', 'ucv_3f_rank',
        'dash_score_median', 'leader_margin', 
        'target_elo_rating', 'horse_age', 'weight_carried',
        'ucv_score_rank', 'prev_ucv_score_rank', 'max_ucv_score_5_rank',
        'elo_rating_rank', 'dash_score_rank',
        'ucv_score_z', 'elo_rating_z', 'weight_carried_diff',
        'jockey_win_rate_100', 'jockey_top3_rate_100', 
        'straight_length', 'elo_diff_from_mean', 'ucv_elo_gap',
        
        # --- 追加特徴量 (馬体重関連) ---
        'horse_weight_num',
        'weight_change_num',
        'carried_weight_ratio',
        # --- アプローチ2 (クロス特徴量) ---
        'weight_change_per_day',
        'is_fatigue_loss',
        # --- Route D (枠順バイアス高解像度版・乗り替わり) ---
        'is_jockey_changed',
        'is_sameday_jockey_change',
        'bracket_win_rate_highres',
        'bracket_top3_rate_highres',
        # --- 追加特徴量 ---
        'interval_days',
        'slope_count', 'woodchip_count',
        # --- 調教密度 (追加) ---
        'slope_per_day', 'woodchip_per_day',
        'slope_total_time_4f_1', 'slope_total_time_4f_2',
        'slope_lap_time_2f_1f_1', 'slope_lap_time_2f_1f_2',
        'slope_lap_time_1f_0m_1', 'slope_lap_time_1f_0m_2',
        'woodchip_total_time_5f_1', 'woodchip_total_time_5f_2',
        'woodchip_total_time_4f_1', 'woodchip_total_time_4f_2',  # 追加 
        'woodchip_total_time_2f_1', 'woodchip_total_time_2f_2',  # 追加
        'woodchip_lap_time_1f_0m_1', 'woodchip_lap_time_1f_0m_2'
    ]

    # トラックコードによる分割
    track_code_int = pd.to_numeric(test_df['track_code'], errors='coerce').fillna(0)
    is_turf_mask = track_code_int.between(10, 22)
    is_dirt_mask = track_code_int.between(23, 29)

    df_turf = test_df[is_turf_mask].copy()
    df_dirt = test_df[is_dirt_mask].copy()

    # それぞれ推論
    df_turf, cat_cols = prepare_features(df_turf)
    actual_features_turf = [col for col in (cat_cols + numeric_features) if col in df_turf.columns]
    booster_turf = lgb.Booster(model_file='lgbm_model_turf.txt')
    df_turf['predict_score'] = booster_turf.predict(df_turf[actual_features_turf])

    df_dirt, _ = prepare_features(df_dirt)
    actual_features_dirt = [col for col in (cat_cols + numeric_features) if col in df_dirt.columns]
    booster_dirt = lgb.Booster(model_file='lgbm_model_dirt.txt')
    df_dirt['predict_score'] = booster_dirt.predict(df_dirt[actual_features_dirt])

    # 結果の結合とランク付け
    result_df = pd.concat([df_turf, df_dirt])
    result_df['score_rank'] = result_df.groupby('race_id')['predict_score'].rank(ascending=False, method='min')

    print(f"\n対象期間: {result_df['race_date'].min()} ～ {result_df['race_date'].max()}")
    print(f"対象レース数: {result_df['race_id'].nunique()} レース")

    # ---------------------------------------------------------
    # メタデータの付与（分析用ラベル）
    # ---------------------------------------------------------
    course_map = {'01':'札幌', '02':'函館', '03':'福島', '04':'新潟', '05':'東京', '06':'中山', '07':'中京', '08':'京都', '09':'阪神', '10':'小倉'}
    result_df['course_name'] = result_df['course_code'].astype(str).str.zfill(2).map(course_map).fillna('その他')
    
    # 距離区分（SMILE基準を一部日本競馬向けにアレンジ）
    bins_dist = [0, 1300, 1899, 2100, 9999]
    labels_dist = ['1_短距離(~1300m)', '2_マイル(1301~1899m)', '3_中距離(1900~2100m)', '4_長距離(2101m~)']
    result_df['dist_category'] = pd.cut(pd.to_numeric(result_df['distance']), bins=bins_dist, labels=labels_dist)
    result_df['track_type'] = np.where(pd.to_numeric(result_df['track_code']).between(10, 22), '芝', 'ダート')

    # モデル本命馬（予測1位）に絞る
    top1_df = result_df[result_df['score_rank'] == 1].copy()

    def print_pivot(index_col, columns_col, value_func, title):
        print(f"\n--- {title} ---")
        pivot = top1_df.groupby([index_col, columns_col], observed=True).apply(value_func, include_groups=False).unstack()
        print(pivot.fillna('-'))

    def calc_hit_rate(x):
        return f"{(len(x[x['is_win']==1]) / len(x) * 100):.1f}%" if len(x) > 0 else "-"
        
    def calc_return_rate(x):
        return f"{(x[x['is_win']==1]['win_odds'].sum() * 100 / len(x)):.1f}%" if len(x) > 0 else "-"

    # ---------------------------------------------------------
    # マトリクス出力: 競馬場 × トラック(芝/ダ) の的中率・回収率
    # ---------------------------------------------------------
    print_pivot('course_name', 'track_type', calc_hit_rate, "競馬場 × トラック別 【的中率(%)】")
    print_pivot('course_name', 'track_type', calc_return_rate, "競馬場 × トラック別 【回収率(%)】")

    # ---------------------------------------------------------
    # マトリクス出力: トラック(芝/ダ) × 距離区分の的中率・回収率
    # ---------------------------------------------------------
    print_pivot('track_type', 'dist_category', calc_hit_rate, "トラック × 距離区分別 【的中率(%)】")
    print_pivot('track_type', 'dist_category', calc_return_rate, "トラック × 距離区分別 【回収率(%)】")

if __name__ == "__main__":
    run_lambdarank_backtest()
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np

import data_loader
import data_splitter

DB_PATH = "C:/Ugaura/sqlite/jra_race.db"

def prepare_features(df):
    """カテゴリカル変数の安全な型変換を行う共通関数"""
    categorical_features = [
        'course_code', 'track_code', 'turf_condition_code', 'dirt_condition_code',
        'bracket_number', 'horse_number', 'sex_code', 'distance', 'prev_running_style',
        'affiliation_code'
    ]
    for col in categorical_features:
        if df[col].dtype.name == 'category':
            if -1 not in df[col].cat.categories:
                df[col] = df[col].cat.add_categories([-1])
            df[col] = df[col].fillna(-1)
        else:
            df[col] = df[col].fillna(-1).astype('category')
    return df, categorical_features

def train_lambdarank_model(df_subset, model_filename, category_name):
    """指定されたデータセットでLambdaRankを学習する関数"""
    print(f"\n========== 【{category_name}】 モデルの学習を開始 ==========")
    
    numeric_features = [
        'pred_lap_diff',
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
        'interval_days',
        'horse_weight_num',
        'weight_change_num',
        'carried_weight_ratio',
        'weight_change_per_day',
        'is_fatigue_loss',
        'is_jockey_changed',
        'is_sameday_jockey_change',
        'bracket_win_rate_highres',
        'bracket_top3_rate_highres',
        'slope_count', 'woodchip_count',
        'slope_per_day', 'woodchip_per_day',
        'slope_total_time_4f_1', 'slope_total_time_4f_2',
        'slope_lap_time_2f_1f_1', 'slope_lap_time_2f_1f_2',
        'slope_lap_time_1f_0m_1', 'slope_lap_time_1f_0m_2',
        'woodchip_total_time_5f_1', 'woodchip_total_time_5f_2',
        'woodchip_total_time_4f_1', 'woodchip_total_time_4f_2',
        'woodchip_total_time_2f_1', 'woodchip_total_time_2f_2',
        'woodchip_lap_time_1f_0m_1', 'woodchip_lap_time_1f_0m_2',
        'slope_best_time_ratio_1', 'slope_best_time_ratio_2',
        'slope_lap_diff_1', 'slope_lap_diff_2',
        'slope_is_acceleration_1', 'slope_is_acceleration_2',
        'woodchip_best_time_ratio_1', 'woodchip_best_time_ratio_2',
        'woodchip_lap_diff_1', 'woodchip_lap_diff_2',
        'woodchip_is_acceleration_1', 'woodchip_is_acceleration_2',
        'prev_corner_4_ratio', 'avg_corner_4_ratio_5',
        'is_local', 'is_small_turn', 'interaction_small_turn_lead',
        'is_steep_slope', 'steep_slope_top3_rate', 'is_short_straight',
        'interaction_short_straight_back', 'interaction_steep_slope_weight',
        'is_true_escape_prev', 'dash_score_z', 'dash_score_in_race_rank',
        'is_fastest_dash_in_race', 'escape_horse_count_in_race',
        'interaction_escape_conflict_lead', 'interaction_escape_conflict_back',
        'interaction_kokura_dirt_true_escape', 'interaction_kokura_dirt_escape_conflict',
        'interaction_kokura_dirt_escape_conflict_back',
        'interaction_nakayama_turf_lead', 'interaction_nakayama_turf_back',
        'distance_change', 'is_promoted',
        'interaction_straight_length_lead', 'interaction_straight_length_back',
        'interaction_chukyo_dirt_lead', 'interaction_chukyo_dirt_back',
        'interaction_kokura_dirt_lead', 'interaction_kokura_dirt_back',
        'interaction_fukushima_dirt_1150_lead', 'interaction_fukushima_dirt_1150_back'
    ]
    
    df_subset, categorical_features = prepare_features(df_subset)
    
    # 存在しないカラムを除外する安全策 (data_loader側の仕様変更吸収用)
    actual_features = [col for col in (categorical_features + numeric_features) if col in df_subset.columns]
    missing_features = set(categorical_features + numeric_features) - set(actual_features)
    if missing_features:
        print(f"Warning: 以下の特徴量はデータフレームに存在しないため除外されます: {missing_features}")
    
    features = actual_features
    target = 'is_win'

    train_df_raw, valid_df_raw, test_df_raw = data_splitter.split_by_time(
        df_subset, train_end_date='2023-12-31', valid_end_date='2024-12-31'
    )

    # LambdaRank用ソート
    train_df = train_df_raw.sort_values('race_id')
    valid_df = valid_df_raw.sort_values('race_id')
    
    train_group = train_df.groupby('race_id').size().values
    valid_group = valid_df.groupby('race_id').size().values

    lgb_train = lgb.Dataset(train_df[features], label=train_df[target], group=train_group, categorical_feature=categorical_features)
    lgb_valid = lgb.Dataset(valid_df[features], label=valid_df[target], group=valid_group, reference=lgb_train, categorical_feature=categorical_features)

    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [1, 3, 5],
        'learning_rate': 0.05,
        'max_depth': 5,
        'num_leaves': 15,
        'min_data_in_leaf': 200,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'random_state': 42,
        'verbose': -1
    }

    callbacks = [lgb.early_stopping(stopping_rounds=100, verbose=True)]

    booster = lgb.train(
        params, lgb_train, num_boost_round=10000,
        valid_sets=[lgb_train, lgb_valid], valid_names=['train', 'valid'], callbacks=callbacks
    )

    booster.save_model(model_filename, num_iteration=booster.best_iteration)
    print(f"完了: {model_filename} を出力しました。")

    # ----- Feature Importance の出力 -----
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': booster.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    
    print("\n--- Feature Importance (Top 15) ---")
    print(importance_df.head(15).to_string(index=False))
    print("-----------------------------------")

    # ----- テストデータを用いたROC-AUCの算出 -----
    if len(test_df_raw) > 0:
        X_test = test_df_raw[features]
        y_test = test_df_raw[target]
        
        # 推論
        y_pred = booster.predict(X_test)
        
        # ROC-AUCの算出
        auc_score = roc_auc_score(y_test, y_pred)
        print(f"\n評価: 【{category_name}】 Test ROC-AUC = {auc_score:.4f}")
    else:
        print(f"\n評価: 【{category_name}】 テストデータが存在しないため、ROC-AUCの評価をスキップしました。")
    # --------------------------------------------------

def train_split_baseline():
    print("1. データのロードと前処理を開始...")
    df = data_loader.load_feature_matrix(DB_PATH)

    print(f"【ノイズ排除前】 レース数: {df['race_id'].nunique()}, サンプル数: {len(df)}")
    df = df[~np.isclose(df['target_elo_rating'], 1000.0, atol=1e-5)].copy()
    valid_race_counts = df.groupby('race_id').size()
    df = df[df['race_id'].isin(valid_race_counts[valid_race_counts >= 2].index)].copy()
    print(f"【ノイズ排除後】 レース数: {df['race_id'].nunique()}, サンプル数: {len(df)}")

    # トラックコードによる物理的な分割 (10番台は芝、20番台はダート・障害等)
    track_code_int = pd.to_numeric(df['track_code'], errors='coerce').fillna(0)
    is_turf_mask = track_code_int.between(10, 22)
    is_dirt_mask = track_code_int.between(23, 29) # 障害(50以上)は基本的に除外されている前提

    df_turf = df[is_turf_mask].copy()
    df_dirt = df[is_dirt_mask].copy()

    # それぞれ独立したモデルとして学習・保存
    train_lambdarank_model(df_turf, 'lgbm_model_turf.txt', '芝(Turf)')
    train_lambdarank_model(df_dirt, 'lgbm_model_dirt.txt', 'ダート(Dirt)')

if __name__ == "__main__":
    train_split_baseline()
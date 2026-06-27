import sys
import sqlite3
import pandas as pd
import numpy as np
import lightgbm as lgb
from data_loader import load_feature_matrix

# Windows環境での文字化け防止のため標準出力をUTF-8に設定
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

TEST_START_DATE = '2025-01-05'
TEST_END_DATE = '2026-04-26'

# フィルタリング設定（Trueでダート中長距離に限定、Falseで全レース投資）
APPLY_FILTER = False

def prepare_features(df):
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

def run_lambdarank_backtest(db_path='C:/Ugaura/sqlite/jra_race.db', calibration_method='platt'):
    print(f"--- 芝/ダート分離 LambdaRank バックテスト開始 (キャリブレーション: {calibration_method}) ---")
    df = load_feature_matrix(db_path)
    # 開始日付と終了日付の両方の条件を満たすデータを抽出
    test_df = df[(df['race_date'] >= TEST_START_DATE) & (df['race_date'] <= TEST_END_DATE)].copy()    

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
        'horse_weight_num',
        'weight_change_num',
        'carried_weight_ratio',
        'weight_change_per_day',
        'is_fatigue_loss',
        'is_jockey_changed',
        'is_sameday_jockey_change',
        'bracket_win_rate_highres',
        'bracket_top3_rate_highres',
        'interval_days',
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

    # 検証データ（2024年）をロードしてフィッティングに使用
    val_df = df[(df['race_date'] >= '2024-01-01') & (df['race_date'] <= '2024-12-31')].copy()
    
    # 芝とダートのモデルをロード
    booster_turf = lgb.Booster(model_file='lgbm_model_turf.txt')
    booster_dirt = lgb.Booster(model_file='lgbm_model_dirt.txt')

    # 検証データの芝とダートのマスク
    val_track_code = pd.to_numeric(val_df['track_code'], errors='coerce').fillna(0)
    is_turf_val = val_track_code.between(10, 22)
    is_dirt_val = val_track_code.between(23, 29)

    val_turf = val_df[is_turf_val].copy()
    val_turf, cat_cols = prepare_features(val_turf)
    actual_features_turf = [col for col in (cat_cols + numeric_features) if col in val_turf.columns]
    val_turf['predict_score'] = booster_turf.predict(val_turf[actual_features_turf])

    val_dirt = val_df[is_dirt_val].copy()
    val_dirt, _ = prepare_features(val_dirt)
    actual_features_dirt = [col for col in (cat_cols + numeric_features) if col in val_dirt.columns]
    val_dirt['predict_score'] = booster_dirt.predict(val_dirt[actual_features_dirt])

    # 馬場状態によるグループ分けを判定する関数
    def get_condition_group(row):
        tc = pd.to_numeric(row['track_code'], errors='coerce')
        if pd.isna(tc):
            return 'dry'
        is_turf = 10 <= tc <= 22
        cond_code = row['turf_condition_code'] if is_turf else row['dirt_condition_code']
        cond_str = str(cond_code).strip()
        return 'dry' if cond_str == '1' else 'wet'

    # キャリブレーションの実行
    if calibration_method == 'platt':
        print("\n--- 確率キャリブレーション（Platt Scaling）のフィッティング ---")
        from sklearn.linear_model import LogisticRegression
        calibrator_turf = LogisticRegression(C=1e9, random_state=42)
        calibrator_turf.fit(val_turf['predict_score'].values.reshape(-1, 1), val_turf['is_win'])
        print(f"芝用キャリブレータ学習完了: 係数={calibrator_turf.coef_[0][0]:.4f}, 切片={calibrator_turf.intercept_[0]:.4f}")

        calibrator_dirt = LogisticRegression(C=1e9, random_state=42)
        calibrator_dirt.fit(val_dirt['predict_score'].values.reshape(-1, 1), val_dirt['is_win'])
        print(f"ダート用キャリブレータ学習完了: 係数={calibrator_dirt.coef_[0][0]:.4f}, 切片={calibrator_dirt.intercept_[0]:.4f}")

    elif calibration_method == 'isotonic':
        print("\n--- 確率キャリブレーション（Isotonic Regression）のフィッティング ---")
        from sklearn.isotonic import IsotonicRegression
        calibrator_turf = IsotonicRegression(out_of_bounds='clip')
        calibrator_turf.fit(val_turf['predict_score'].values, val_turf['is_win'])
        print("芝用等張回帰キャリブレータ学習完了")

        calibrator_dirt = IsotonicRegression(out_of_bounds='clip')
        calibrator_dirt.fit(val_dirt['predict_score'].values, val_dirt['is_win'])
        print("ダート用等張回帰キャリブレータ学習完了")

    elif calibration_method == 'group_platt':
        print("\n--- 確率キャリブレーション（Group-based Platt Scaling: 馬場状態別）のフィッティング ---")
        from sklearn.linear_model import LogisticRegression
        val_turf['cond_group'] = val_turf.apply(get_condition_group, axis=1)
        val_dirt['cond_group'] = val_dirt.apply(get_condition_group, axis=1)

        calibrators_turf = {}
        for g in ['dry', 'wet']:
            sub = val_turf[val_turf['cond_group'] == g]
            if len(sub) > 0:
                lr = LogisticRegression(C=1e9, random_state=42)
                lr.fit(sub['predict_score'].values.reshape(-1, 1), sub['is_win'])
                calibrators_turf[g] = lr
                print(f"芝 ({g}) 学習完了: 係数={lr.coef_[0][0]:.4f}, 切片={lr.intercept_[0]:.4f}")

        calibrators_dirt = {}
        for g in ['dry', 'wet']:
            sub = val_dirt[val_dirt['cond_group'] == g]
            if len(sub) > 0:
                lr = LogisticRegression(C=1e9, random_state=42)
                lr.fit(sub['predict_score'].values.reshape(-1, 1), sub['is_win'])
                calibrators_dirt[g] = lr
                print(f"ダート ({g}) 学習完了: 係数={lr.coef_[0][0]:.4f}, 切片={lr.intercept_[0]:.4f}")

    # テストデータの予測とロジット算出 / 確率算出
    track_code_int = pd.to_numeric(test_df['track_code'], errors='coerce').fillna(0)
    is_turf_mask = track_code_int.between(10, 22)
    is_dirt_mask = track_code_int.between(23, 29)

    df_turf = test_df[is_turf_mask].copy()
    df_dirt = test_df[is_dirt_mask].copy()

    df_turf, cat_cols = prepare_features(df_turf)
    actual_features_turf = [col for col in (cat_cols + numeric_features) if col in df_turf.columns]
    df_turf['predict_score'] = booster_turf.predict(df_turf[actual_features_turf])

    df_dirt, _ = prepare_features(df_dirt)
    actual_features_dirt = [col for col in (cat_cols + numeric_features) if col in df_dirt.columns]
    df_dirt['predict_score'] = booster_dirt.predict(df_dirt[actual_features_dirt])

    if calibration_method == 'platt':
        df_turf['calibrated_logit'] = calibrator_turf.decision_function(df_turf['predict_score'].values.reshape(-1, 1))
        df_dirt['calibrated_logit'] = calibrator_dirt.decision_function(df_dirt['predict_score'].values.reshape(-1, 1))
        result_df = pd.concat([df_turf, df_dirt])

        def softmax_logit(x):
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()
        result_df['estimated_prob'] = result_df.groupby('race_id')['calibrated_logit'].transform(softmax_logit)

    elif calibration_method == 'isotonic':
        df_turf['calibrated_prob'] = calibrator_turf.predict(df_turf['predict_score'].values)
        df_dirt['calibrated_prob'] = calibrator_dirt.predict(df_dirt['predict_score'].values)
        result_df = pd.concat([df_turf, df_dirt])

        prob_sum = result_df.groupby('race_id')['calibrated_prob'].transform('sum')
        horse_count = result_df.groupby('race_id')['horse_id'].transform('count')
        result_df['estimated_prob'] = np.where(prob_sum > 1e-6, result_df['calibrated_prob'] / prob_sum, 1.0 / horse_count)

    elif calibration_method == 'group_platt':
        df_turf['cond_group'] = df_turf.apply(get_condition_group, axis=1)
        df_dirt['cond_group'] = df_dirt.apply(get_condition_group, axis=1)

        df_turf['calibrated_logit'] = 0.0
        for g in ['dry', 'wet']:
            mask = df_turf['cond_group'] == g
            if mask.any() and g in calibrators_turf:
                df_turf.loc[mask, 'calibrated_logit'] = calibrators_turf[g].decision_function(df_turf.loc[mask, 'predict_score'].values.reshape(-1, 1))

        df_dirt['calibrated_logit'] = 0.0
        for g in ['dry', 'wet']:
            mask = df_dirt['cond_group'] == g
            if mask.any() and g in calibrators_dirt:
                df_dirt.loc[mask, 'calibrated_logit'] = calibrators_dirt[g].decision_function(df_dirt.loc[mask, 'predict_score'].values.reshape(-1, 1))

        result_df = pd.concat([df_turf, df_dirt])

        def softmax_logit(x):
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()
        result_df['estimated_prob'] = result_df.groupby('race_id')['calibrated_logit'].transform(softmax_logit)

    result_df['score_rank'] = result_df.groupby('race_id')['predict_score'].rank(ascending=False, method='min')
    result_df['expected_value'] = result_df['estimated_prob'] * result_df['win_odds']

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

    def get_ev_threshold(course, track):
        low_recovery_races = {
            ('札幌', 'ダート'),
            ('福島', '芝'),
            ('新潟', '芝'),
            ('東京', 'ダート'),
            ('京都', 'ダート'),
            ('京都', '芝'),
            ('小倉', 'ダート'),
            ('小倉', '芝'),
        }
        medium_low_recovery_races = {
            ('札幌', '芝'),
            ('函館', 'ダート'),
            ('中山', '芝'),
            ('阪神', '芝'),
        }
        if (course, track) in low_recovery_races:
            return 1.30
        elif (course, track) in medium_low_recovery_races:
            return 1.15
        else:
            return 1.00

    # コースごとのしきい値を適用
    result_df['ev_threshold'] = result_df.apply(lambda row: get_ev_threshold(row['course_name'], row['track_type']), axis=1)

    # 購入条件の設定：
    # 1. 予測1位であること
    # 2. 期待値がそのコースのしきい値以上であること
    buy_mask = (result_df['score_rank'] == 1) & (result_df['expected_value'] >= result_df['ev_threshold'])

    if APPLY_FILTER:
        # フィルタリング条件：ダートかつ距離1900m以上（中長距離）
        filter_mask = (result_df['track_type'] == 'ダート') & (pd.to_numeric(result_df['distance']) >= 1900)
        buy_mask = buy_mask & filter_mask
        print("\n[情報] フィルタリングが有効です：投資対象を「ダート中長距離（1900m以上）」に限定します。")
    else:
        print("\n[情報] フィルタリングは無効です：全コースを投資対象とします。")

    buy_df = result_df[buy_mask].copy()

    print(f"\n【購入シミュレーション】")
    print(f"予測1位の全頭数: {len(result_df[result_df['score_rank'] == 1])}")
    print(f"うち、コース別しきい値以上の購入対象頭数: {len(buy_df)}")

    def print_pivot(index_col, columns_col, value_func, title):
        print(f"\n--- {title} ---")
        pivot = buy_df.groupby([index_col, columns_col], observed=True).apply(value_func, include_groups=False).unstack()
        print(pivot.fillna('-'))

    def calc_hit_rate(x):
        return f"{(len(x[x['is_win']==1]) / len(x) * 100):.1f}%" if len(x) > 0 else "-"
        
    def calc_return_rate(x):
        return f"{(x[x['is_win']==1]['win_odds'].sum() * 100 / len(x)):.1f}%" if len(x) > 0 else "-"

    # ---------------------------------------------------------
    # 競馬場 × トラック別の詳細集計 (レース数、的中数、的中率、回収率)
    # ---------------------------------------------------------
    print("\n--- 競馬場 × トラック別 詳細集計結果 ---")
    summary_list = []
    grouped = buy_df.groupby(['course_name', 'track_type'], observed=True)
    for (course, track), group in grouped:
        races = len(group)
        hits = len(group[group['is_win'] == 1])
        hit_rate = (hits / races * 100) if races > 0 else 0.0
        return_rate = (group[group['is_win'] == 1]['win_odds'].sum() / races * 100) if races > 0 else 0.0
        summary_list.append({
            '競馬場': course,
            'トラック': track,
            'レース数': races,
            '的中数': hits,
            '的中率': f"{hit_rate:.1f}%",
            '回収率': f"{return_rate:.1f}%",
            '_sort_return': return_rate
        })
    if summary_list:
        summary_df = pd.DataFrame(summary_list)
        course_order = {'札幌':1, '函館':2, '福島':3, '新潟':4, '東京':5, '中山':6, '中京':7, '京都':8, '阪神':9, '小倉':10, 'その他':11}
        summary_df['course_order'] = summary_df['競馬場'].map(course_order).fillna(99)
        summary_df = summary_df.sort_values(by=['course_order', 'トラック']).drop(columns=['course_order', '_sort_return'])
        print(summary_df.to_string(index=False))
    else:
        print("購入対象データがありません。")

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

    # ---------------------------------------------------------
    # 低回収・高回収グループの傾向分析
    # ---------------------------------------------------------
    def get_group_label(row):
        c = row['course_name']
        t = row['track_type']
        if (c == '福島' and t == '芝') or \
           (c == '札幌' and t == 'ダート') or \
           (c == '小倉' and t == 'ダート') or \
           (c == '東京' and t == 'ダート') or \
           (c == '京都' and t == 'ダート') or \
           (c == '阪神' and t == 'ダート'):
            return '低回収グループ'
        elif (c == '札幌' and t == '芝') or \
             (c == '函館' and t == '芝') or \
             (c == '新潟' and t == '芝') or \
             (c == '中京' and t == 'ダート') or \
             (c == '京都' and t == '芝'):
            return '高回収グループ'
        else:
            return 'その他'

    print("\n--- 低回収・高回収グループの傾向分析 ---")
    
    # 的中オッズランク（人気順）の算出。win_oddsが低い順
    result_df['odds_rank'] = result_df.groupby('race_id')['win_odds'].rank(ascending=True, method='min')
    
    # buy_dfにodds_rankをマージ（重複を避けるために一意のキーでマージ、あるいはそのまま参照）
    # buy_dfに直接 result_dfの odds_rank を入れる
    buy_df = buy_df.merge(result_df[['race_id', 'horse_number', 'odds_rank']], on=['race_id', 'horse_number'], how='left')
    buy_df['analysis_group'] = buy_df.apply(get_group_label, axis=1)
    
    # 各グループの基礎統計
    # pandasのagg内でカスタム関数を使う場合、Seriesに対してのインデックスのずれに注意して計算
    def get_hit_rate(s):
        return f"{(s == 1).mean() * 100:.1f}%" if len(s) > 0 else "0.0%"
        
    def get_return_rate(df_sub):
        # buy_dfから直接絞り込んで計算
        idx = df_sub.index
        subset = buy_df.loc[idx]
        races = len(subset)
        if races == 0:
            return "0.0%"
        ret = subset[subset['is_win'] == 1]['win_odds'].sum() / races * 100
        return f"{ret:.1f}%"

    # groupby.applyを使って安全に集計
    agg_list = []
    for name, group in buy_df.groupby('analysis_group'):
        if name not in ['低回収グループ', '高回収グループ']:
            continue
        races = len(group)
        hits = len(group[group['is_win'] == 1])
        hit_rate = (hits / races * 100) if races > 0 else 0.0
        return_rate = (group[group['is_win'] == 1]['win_odds'].sum() / races * 100) if races > 0 else 0.0
        fav1_rate = (group['odds_rank'] == 1).mean() * 100 if races > 0 else 0.0
        
        agg_list.append({
            'グループ': name,
            '購入頭数': races,
            '的中数': hits,
            '的中率': f"{hit_rate:.1f}%",
            '回収率': f"{return_rate:.1f}%",
            '平均オッズ': f"{group['win_odds'].mean():.2f}",
            '中央値オッズ': f"{group['win_odds'].median():.1f}",
            '平均推定勝率': f"{(group['estimated_prob'].mean() * 100):.1f}%",
            '平均期待値': f"{group['expected_value'].mean():.2f}",
            '1番人気率': f"{fav1_rate:.1f}%"
        })
        
    agg_df = pd.DataFrame(agg_list)
    print("\n[基礎統計の比較]")
    print(agg_df.to_string(index=False))

    # 主要特徴量の比較
    key_features = [
        'jockey_top3_rate_100', 'ucv_score_z', 'avg_ucv_3f_5', 
        'dash_score_median', 'jockey_win_rate_100', 'horse_age', 'weight_carried'
    ]
    
    print("\n[主要特徴量の平均値比較（トラックごと）]")
    for track in ['芝', 'ダート']:
        track_buy = buy_df[buy_df['track_type'] == track]
        if len(track_buy) == 0:
            continue
        print(f"\n--- {track} ---")
        features_to_compare = [f for f in key_features if f in track_buy.columns]
        feat_comparison = track_buy.groupby('analysis_group')[features_to_compare].mean().reindex(['低回収グループ', '高回収グループ'])
        print(feat_comparison.to_string())

    return result_df, buy_df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--calibration', type=str, default='platt', choices=['platt', 'isotonic', 'group_platt'],
                        help='Calibration method: platt, isotonic, group_platt')
    args = parser.parse_args()
    
    run_lambdarank_backtest(calibration_method=args.calibration)
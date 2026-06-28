import argparse
import sys
import sqlite3
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from data_loader import load_feature_matrix
from realtime_loader import download_and_update_realtime

DB_PATH = "C:/Ugaura/sqlite/jra_race.db"

def get_condition_group(row):
    tc = pd.to_numeric(row['track_code'], errors='coerce')
    if pd.isna(tc):
        return 'dry'
    is_turf = 10 <= tc <= 22
    cond_code = row['turf_condition_code'] if is_turf else row['dirt_condition_code']
    cond_str = str(cond_code).strip()
    return 'dry' if cond_str == '1' else 'wet'

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

def detect_track_bias(cursor, year, month_day, course_code, is_turf):
    """
    当日の前半レース結果からトラックバイアス（脚質・枠順）を検出する
    """
    track_min, track_max = (10, 22) if is_turf else (23, 29)
    
    query = """
        SELECT h.final_order, h.corner_4_order, h.bracket_number,
               (h.year || h.month_day || h.course_code || h.times || h.day || h.race_number) AS race_id
        FROM horse_race_info h
        JOIN race_detail r ON (r.year = h.year AND r.month_day = h.month_day AND r.course_code = h.course_code AND r.times = h.times AND r.day = h.day AND r.race_number = h.race_number)
        WHERE h.year = ? AND h.month_day = ? AND h.course_code = ? AND CAST(r.track_code AS INTEGER) BETWEEN ? AND ?
          AND h.final_order IN ('01', '02', '03', '1', '2', '3')
          AND h.corner_4_order IS NOT NULL AND h.corner_4_order != '' AND h.corner_4_order != '00'
    """
    cursor.execute(query, (year, month_day, course_code, track_min, track_max))
    rows = cursor.fetchall()
    
    if not rows:
        return 0.0, 0.0
        
    df_tb = pd.DataFrame(rows, columns=['final_order', 'corner_4_order', 'bracket_number', 'race_id'])
    df_tb['corner_4_order'] = pd.to_numeric(df_tb['corner_4_order'], errors='coerce')
    df_tb['bracket_number'] = pd.to_numeric(df_tb['bracket_number'], errors='coerce')
    
    race_counts = {}
    for rid in df_tb['race_id'].unique():
        cursor.execute(
            "SELECT COUNT(*) FROM horse_race_info WHERE (year || month_day || course_code || times || day || race_number) = ?",
            (rid,)
        )
        race_counts[rid] = cursor.fetchone()[0]
        
    df_tb['race_horse_count'] = df_tb['race_id'].map(race_counts)
    df_tb['corner_ratio'] = df_tb['corner_4_order'] / df_tb['race_horse_count']
    
    avg_corner_ratio = df_tb['corner_ratio'].mean()
    avg_bracket = df_tb['bracket_number'].mean()
    
    bias_pace = 0.0
    bias_bracket = 0.0
    
    if avg_corner_ratio <= 0.42:
        bias_pace = 1.0
    elif avg_corner_ratio >= 0.58:
        bias_pace = -1.0
        
    if avg_bracket <= 3.8:
        bias_bracket = -1.0
    elif avg_bracket >= 5.2:
        bias_bracket = 1.0
        
    return bias_pace, bias_bracket

def predict_single_race(race_df, df_all, cursor, year, month_day, course_code, times, day, race_number, turf_booster, dirt_booster):
    """
    1つのレースの予測およびEV再計算を実行して結果を表示する
    """
    track_code_int = pd.to_numeric(race_df['track_code'].iloc[0], errors='coerce')
    is_turf = 10 <= track_code_int <= 22
    track_type = '芝' if is_turf else 'ダート'
    
    # トラックバイアス検出
    bias_pace, bias_bracket = detect_track_bias(cursor, year, month_day, course_code, is_turf)
    
    # キャリブレータのフィッティング (2024年データ使用)
    val_df = df_all[(df_all['race_date'] >= '2024-01-01') & (df_all['race_date'] <= '2024-12-31')].copy()
    
    booster = turf_booster if is_turf else dirt_booster
    
    numeric_features = [
        'pred_lap_diff', 'pace_score_top3', 'prev_ucv_score', 'max_ucv_score_5', 'avg_ucv_score_5', 
        'avg_ucv_3f_5', 'max_ucv_3f_5', 'ucv_3f_gap_avg_5', 'ucv_3f_rank', 'dash_score_median', 'leader_margin', 
        'target_elo_rating', 'horse_age', 'weight_carried', 'ucv_score_rank', 'prev_ucv_score_rank', 'max_ucv_score_5_rank',
        'elo_rating_rank', 'dash_score_rank', 'ucv_score_z', 'elo_rating_z', 'weight_carried_diff', 'jockey_win_rate_100', 
        'jockey_top3_rate_100', 'straight_length', 'elo_diff_from_mean', 'ucv_elo_gap', 'interval_days', 'horse_weight_num',
        'weight_change_num', 'carried_weight_ratio', 'weight_change_per_day', 'is_fatigue_loss', 'is_jockey_changed',
        'is_sameday_jockey_change', 'bracket_win_rate_highres', 'bracket_top3_rate_highres', 'slope_count', 'woodchip_count',
        'slope_per_day', 'woodchip_per_day', 'slope_total_time_4f_1', 'slope_total_time_4f_2', 'slope_lap_time_2f_1f_1', 
        'slope_lap_time_2f_1f_2', 'slope_lap_time_1f_0m_1', 'slope_lap_time_1f_0m_2', 'woodchip_total_time_5f_1', 
        'woodchip_total_time_5f_2', 'woodchip_total_time_4f_1', 'woodchip_total_time_4f_2', 'woodchip_total_time_2f_1', 
        'woodchip_total_time_2f_2', 'woodchip_lap_time_1f_0m_1', 'woodchip_lap_time_1f_0m_2', 'slope_best_time_ratio_1', 
        'slope_best_time_ratio_2', 'slope_lap_diff_1', 'slope_lap_diff_2', 'slope_is_acceleration_1', 'slope_is_acceleration_2',
        'woodchip_best_time_ratio_1', 'woodchip_best_time_ratio_2', 'woodchip_lap_diff_1', 'woodchip_lap_diff_2', 
        'woodchip_is_acceleration_1', 'woodchip_is_acceleration_2', 'prev_corner_4_ratio', 'avg_corner_4_ratio_5',
        'is_local', 'is_small_turn', 'interaction_small_turn_lead', 'is_steep_slope', 'steep_slope_top3_rate', 'is_short_straight',
        'interaction_short_straight_back', 'interaction_steep_slope_weight', 'is_true_escape_prev', 'dash_score_z', 
        'dash_score_in_race_rank', 'is_fastest_dash_in_race', 'escape_horse_count_in_race', 'interaction_escape_conflict_lead', 
        'interaction_escape_conflict_back', 'interaction_kokura_dirt_true_escape', 'interaction_kokura_dirt_escape_conflict',
        'interaction_kokura_dirt_escape_conflict_back', 'interaction_nakayama_turf_lead', 'interaction_nakayama_turf_back',
        'distance_change', 'is_promoted', 'interaction_straight_length_lead', 'interaction_straight_length_back',
        'interaction_chukyo_dirt_lead', 'interaction_chukyo_dirt_back', 'interaction_kokura_dirt_lead', 'interaction_kokura_dirt_back',
        'interaction_fukushima_dirt_1150_lead', 'interaction_fukushima_dirt_1150_back'
    ]
    
    val_track_code = pd.to_numeric(val_df['track_code'], errors='coerce').fillna(0)
    is_turf_val = val_track_code.between(10, 22)
    val_subset = val_df[is_turf_val].copy() if is_turf else val_df[~is_turf_val].copy()
    
    val_subset, cat_cols = prepare_features(val_subset)
    actual_features = [col for col in (cat_cols + numeric_features) if col in val_subset.columns]
    val_subset['predict_score'] = booster.predict(val_subset[actual_features])
    
    val_subset['cond_group'] = val_subset.apply(get_condition_group, axis=1)
    
    target_cond = get_condition_group(race_df.iloc[0])
    
    sub_val = val_subset[val_subset['cond_group'] == target_cond]
    calibrator = LogisticRegression(C=1e9, random_state=42)
    calibrator.fit(sub_val['predict_score'].values.reshape(-1, 1), sub_val['is_win'])
    
    # 対象レースの予測
    race_df, cat_cols = prepare_features(race_df)
    actual_features = [col for col in (cat_cols + numeric_features) if col in race_df.columns]
    race_df['predict_score'] = booster.predict(race_df[actual_features])
    
    race_df['calibrated_logit'] = calibrator.decision_function(race_df['predict_score'].values.reshape(-1, 1))
    
    e_x = np.exp(race_df['calibrated_logit'] - np.max(race_df['calibrated_logit']))
    race_df['estimated_prob'] = e_x / e_x.sum()
    
    race_df['expected_value'] = race_df['estimated_prob'] * race_df['win_odds']
    
    # TB補正適用
    race_df['tb_adjusted_ev'] = race_df['expected_value']
    
    tb_msg = []
    if bias_pace == 1.0:
        race_df.loc[race_df['avg_corner_4_ratio_5'] <= 0.4, 'tb_adjusted_ev'] += 0.05
        tb_msg.append("先行馬(4角比率<=0.4) EV+0.05")
    elif bias_pace == -1.0:
        race_df.loc[race_df['avg_corner_4_ratio_5'] >= 0.6, 'tb_adjusted_ev'] += 0.05
        tb_msg.append("差し馬(4角比率>=0.6) EV+0.05")
        
    bracket_num = pd.to_numeric(race_df['bracket_number'], errors='coerce')
    if bias_bracket == -1.0:
        race_df.loc[bracket_num <= 3, 'tb_adjusted_ev'] += 0.03
        tb_msg.append("内枠馬(枠番<=3) EV+0.03")
    elif bias_bracket == 1.0:
        race_df.loc[bracket_num >= 6, 'tb_adjusted_ev'] += 0.03
        tb_msg.append("外枠馬(枠番>=6) EV+0.03")
        
    course_map = {'01':'札幌', '02':'函館', '03':'福島', '04':'新潟', '05':'東京', '06':'中山', '07':'中京', '08':'京都', '09':'阪神', '10':'小倉'}
    course_name = course_map.get(course_code, 'その他')
    
    threshold = get_ev_threshold(course_name, track_type)
    
    race_df['realtime_rank'] = race_df['predict_score'].rank(ascending=False, method='min')
    
    # 正式馬名を取得
    db_horse_names = {}
    try:
        cursor.execute(
            """
            SELECT CAST(horse_number AS INTEGER), horse_name 
            FROM horse_race_info 
            WHERE year=? AND month_day=? AND course_code=? AND times=? AND day=? AND race_number=?
            """,
            (year, month_day, course_code, times, day, race_number)
        )
        for hn, name in cursor.fetchall():
            db_horse_names[hn] = name
    except Exception:
        pass

    tb_status = " | ".join(tb_msg) if tb_msg else "補正なし"
    print("\n================== 推論結果 ==================")
    print(f"対象レース: {course_name} {int(race_number)}R ({track_type}) | 馬場状態グループ: {target_cond} | TB補正: {tb_status}")
    print(f"EV購入しきい値: {threshold:.2f}")
    print("----------------------------------------------")
    
    results_to_show = race_df.sort_values('realtime_rank')
    for _, row in results_to_show.iterrows():
        b_num = int(pd.to_numeric(row['bracket_number'], errors='coerce'))
        h_num = int(pd.to_numeric(row['horse_number'], errors='coerce'))
        h_name = db_horse_names.get(h_num, "不明")
        
        rec_status = ""
        if row['realtime_rank'] == 1 and row['tb_adjusted_ev'] >= threshold:
            rec_status = "★購入推奨馬"
            
        out_line = f"予測{int(row['realtime_rank'])}位 | {b_num}枠{h_num}番: {h_name} | 単勝オッズ: {row['win_odds']:.1f}倍 | 予測勝率: {row['estimated_prob']*100:.1f}% | 最終EV: {row['tb_adjusted_ev']:.2f} (生EV: {row['expected_value']:.2f}) {rec_status}"
        print(out_line.encode('cp932', errors='replace').decode('cp932'))
    print("==============================================\n")

def main():
    parser = argparse.ArgumentParser(description="JRA-VAN JV-Link直接連携リアルタイム予測スクリプト")
    parser.add_argument('--date', type=str, default=None, help="予測対象日付 (例: 20260426)")
    parser.add_argument('--race_id', type=str, default=None, help="netkeibaの12桁レースID (例: 202605020411)")
    parser.add_argument('--race_number', type=str, default=None, help="特定レース番号のみ予測する場合 (例: 11)")
    args = parser.parse_args()
    
    date_str = args.date
    target_race_num = args.race_number
    
    # race_idから日付を抽出 (指定が無い場合)
    if not date_str and args.race_id:
        # 12桁ID: YYYYCCxxYYZZ
        # year, course_code, times, day をパース
        year = args.race_id[0:4]
        course_code = args.race_id[4:6]
        times = args.race_id[6:8]
        day = args.race_id[8:10]
        target_race_num = args.race_id[10:12]
        
        # DBから日付を取得
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT DISTINCT month_day FROM race_detail WHERE year=? AND course_code=? AND times=? AND day=?",
            (year, course_code, times, day)
        )
        row = cursor.fetchone()
        conn.close()
        if row:
            date_str = year + row[0]
            print(f"[情報] race_idから日付を特定しました: {date_str}")
        else:
            print("[エラー] 指定されたrace_idの開催日がデータベースに見つかりません。")
            return

    if not date_str:
        print("[エラー] 予測日付を指定してください (--date YYYYMMDD)")
        return
        
    # 1. JV-Linkから最新速報データをロード
    print("--- 1. JRA-VAN JV-Link からのデータインポート処理 ---")
    if not download_and_update_realtime(DB_PATH, date_str):
        print("[エラー] JRA-VANからのリアルタイムインポートに失敗しました。処理を即時停止します。")
        sys.exit(1)

    # 2. 特徴量のロード
    print("--- 2. 特徴量のロード中 (※少し時間がかかります) ---")
    df_all = load_feature_matrix(DB_PATH)
    
    # 対象日のレースに絞り込む
    # race_date は "YYYY-MM-DD" フォーマット
    formatted_date = f"{date_str[0:4]}-{date_str[4:6]}-{date_str[6:8]}"
    day_df = df_all[df_all['race_date'] == formatted_date].copy()
    
    if day_df.empty:
        print(f"[エラー] 該当日の特徴量データがデータベースからロードできませんでした ({formatted_date})")
        return
        
    # モデルの事前ロード
    print("--- 3. LGBMモデルのロード ---")
    turf_booster = lgb.Booster(model_file='lgbm_model_turf.txt')
    dirt_booster = lgb.Booster(model_file='lgbm_model_dirt.txt')
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 当日内の各レースを順次予測
    # race_id ごとにグループ化
    race_ids = day_df['race_id'].unique()
    race_ids.sort()
    
    for db_race_id in race_ids:
        # レースIDからレース番号（末尾2桁）を取得
        race_num_str = db_race_id[-2:]
        if target_race_num and race_num_str != target_race_num.zfill(2):
            continue
            
        race_df = day_df[day_df['race_id'] == db_race_id].copy()
        
        # キー項目抽出
        year = db_race_id[0:4]
        month_day = db_race_id[4:8]
        course_code = db_race_id[8:10]
        times = db_race_id[10:12]
        day = db_race_id[12:14]
        race_number = db_race_id[14:16]
        
        try:
            predict_single_race(
                race_df, df_all, cursor, 
                year, month_day, course_code, times, day, race_number, 
                turf_booster, dirt_booster
            )
        except Exception as e:
            print(f"[エラー] レース {int(race_number)}R の予測中にエラーが発生しました: {e}")
            
    conn.close()

if __name__ == "__main__":
    main()

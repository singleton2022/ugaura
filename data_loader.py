import sqlite3
import pandas as pd
import numpy as np
import time
import duckdb
from datetime import datetime

def load_feature_matrix(db_path):
    def log(msg):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

    log("[1/6] メインSQLクエリの実行開始 (※Window関数を含むためDB容量によっては数分かかります)")
    start_time = time.time()
    
    query = """
    WITH horse_race_history AS (
        SELECT 
            rd.year || rd.month_day || rd.course_code || rd.times || rd.day || rd.race_number AS race_id,
            rd.year || '-' || substr(rd.month_day, 1, 2) || '-' || substr(rd.month_day, 3, 2) AS race_date,
            -- レース間隔算出用の前走日付取得（ここで計算しても問題ありません）
            LAG(rd.year || '-' || substr(rd.month_day, 1, 2) || '-' || substr(rd.month_day, 3, 2)) OVER (PARTITION BY hri.blood_reg_number ORDER BY rd.year || rd.month_day || rd.course_code || rd.times || rd.day || rd.race_number ASC) AS prev_race_date,
            -- 前走からの距離変化・昇級判定用の前走データ取得
            LAG(rd.distance) OVER (PARTITION BY hri.blood_reg_number ORDER BY rd.year || rd.month_day || rd.course_code || rd.times || rd.day || rd.race_number ASC) AS prev_distance,
            LAG(rd.cond_code_youngest) OVER (PARTITION BY hri.blood_reg_number ORDER BY rd.year || rd.month_day || rd.course_code || rd.times || rd.day || rd.race_number ASC) AS prev_cond_code_youngest,
            hri.blood_reg_number AS horse_id,
            ucv.ucv_score,
            u3f.ucv_3f_score
        FROM race_detail rd
        JOIN horse_race_info hri 
            ON  rd.year = hri.year AND rd.month_day = hri.month_day AND rd.course_code = hri.course_code 
            AND rd.times = hri.times AND rd.day = hri.day AND rd.race_number = hri.race_number
        -- 【修正1】LEFT JOIN に変更し、未来のレースを保持する
        LEFT JOIN horse_race_ucv ucv 
            ON  (rd.year || rd.month_day || rd.course_code || rd.times || rd.day || rd.race_number) = ucv.race_id 
            AND hri.horse_number = ucv.horse_number
        -- 【修正2】結合条件を ucv 依存から rd 依存に変更（ucvがNULLでも結合できるようにする）
        LEFT JOIN horse_race_ucv_3f u3f
            ON  (rd.year || rd.month_day || rd.course_code || rd.times || rd.day || rd.race_number) = u3f.race_id 
            AND hri.horse_number = u3f.horse_number
        WHERE hri.abnormality_code IN ('0', '7')
          -- 【修正3】過去の無効レース（地方・障害など）は排除し、未来のレース（着順未定）のみ通す
          AND (
              ucv.ucv_score IS NOT NULL 
              OR TRIM(IFNULL(hri.final_order, '')) IN ('', '0', '00')
          )
    ),
    ucv_rolling AS (
        SELECT 
            race_id,
            horse_id,
            LAG(ucv_score, 1) OVER (PARTITION BY horse_id ORDER BY race_date ASC, race_id ASC) AS prev_ucv_score,
            MAX(ucv_score) OVER (PARTITION BY horse_id ORDER BY race_date ASC, race_id ASC ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS max_ucv_score_5,
            AVG(ucv_score) OVER (PARTITION BY horse_id ORDER BY race_date ASC, race_id ASC ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS avg_ucv_score_5,
            MAX(ucv_3f_score) OVER (PARTITION BY horse_id ORDER BY race_date ASC, race_id ASC ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS max_ucv_3f_5,
            AVG(ucv_3f_score) OVER (PARTITION BY horse_id ORDER BY race_date ASC, race_id ASC ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS avg_ucv_3f_5
        FROM horse_race_history
    ),
    jockey_history AS (
        SELECT 
            rd.year || rd.month_day || rd.course_code || rd.times || rd.day || rd.race_number AS race_id,
            rd.year || '-' || substr(rd.month_day, 1, 2) || '-' || substr(rd.month_day, 3, 2) AS race_date,
            hri.jockey_code,
            CASE WHEN CAST(hri.final_order AS INTEGER) = 1 THEN 1.0 ELSE 0.0 END AS is_win,
            CASE WHEN CAST(hri.final_order AS INTEGER) <= 3 THEN 1.0 ELSE 0.0 END AS is_top3
        FROM race_detail rd
        JOIN horse_race_info hri 
            ON  rd.year = hri.year AND rd.month_day = hri.month_day AND rd.course_code = hri.course_code 
            AND rd.times = hri.times AND rd.day = hri.day AND rd.race_number = hri.race_number
        WHERE hri.abnormality_code IN ('0', '7')
    ),
    jockey_rolling AS (
        SELECT 
            race_id,
            jockey_code,
            AVG(is_win) OVER (PARTITION BY jockey_code ORDER BY race_date ASC, race_id ASC ROWS BETWEEN 100 PRECEDING AND 1 PRECEDING) AS jockey_win_rate_100,
            AVG(is_top3) OVER (PARTITION BY jockey_code ORDER BY race_date ASC, race_id ASC ROWS BETWEEN 100 PRECEDING AND 1 PRECEDING) AS jockey_top3_rate_100
        FROM jockey_history
    ),
    formatted_races AS (
        SELECT 
            rd.year || rd.month_day || rd.course_code || rd.times || rd.day || rd.race_number AS race_id,
            rd.year || '-' || substr(rd.month_day, 1, 2) || '-' || substr(rd.month_day, 3, 2) AS race_date,
            rd.year, rd.month_day, rd.course_code, rd.times, rd.day, rd.race_number,
            rd.distance, rd.track_code, rd.turf_condition_code, rd.dirt_condition_code,
            rd.weather_code, rd.grade_code, rd.race_type_code, rd.weight_type_code, rd.cond_code_youngest
        FROM race_detail rd
        WHERE rd.year >= '2015'
          AND CAST(rd.track_code AS INTEGER) < 51 
          AND CAST(rd.course_code AS INTEGER) BETWEEN 1 AND 10
    ),
    target_data AS (
        SELECT 
            fr.race_id,
            fr.race_date,
            hrh.prev_race_date,
            hrh.prev_distance,
            hrh.prev_cond_code_youngest,
            hri.blood_reg_number AS horse_id,
            hri.horse_number,
            hri.bracket_number,
            hri.sex_code,
            hri.horse_age,
            hri.weight_carried,
            hri.affiliation_code,
            hri.jockey_code,
            hri.win_odds,
            hri.horse_weight,
            hri.weight_change_sign,
            hri.weight_change,
            hri.running_style,
            hri.corner_4_order,
            fr.course_code,
            fr.distance,
            fr.track_code,
            fr.turf_condition_code,
            fr.dirt_condition_code,
            fr.weather_code,
            fr.grade_code,
            fr.race_type_code,
            fr.weight_type_code,
            fr.cond_code_youngest,
            hri.jockey_code_before,
            CASE WHEN CAST(hri.final_order AS INTEGER) = 1 THEN 1 ELSE 0 END AS is_win,
            CASE WHEN CAST(hri.final_order AS INTEGER) <= 3 THEN 1 ELSE 0 END AS is_top3
        FROM formatted_races fr
        JOIN horse_race_info hri 
            ON  fr.year = hri.year AND fr.month_day = hri.month_day AND fr.course_code = hri.course_code 
            AND fr.times = hri.times AND fr.day = hri.day AND fr.race_number = hri.race_number
        LEFT JOIN horse_race_history hrh 
            ON fr.race_id = hrh.race_id AND hri.blood_reg_number = hrh.horse_id
        WHERE hri.abnormality_code IN ('0', '7')
    )
    SELECT 
        t.*,
        ur.prev_ucv_score, ur.max_ucv_score_5, ur.avg_ucv_score_5,
        ur.max_ucv_3f_5, ur.avg_ucv_3f_5,
        jr.jockey_win_rate_100, jr.jockey_top3_rate_100,
        ds.dash_score_median, pp.pace_score_top3, pp.leader_margin,
        rt.elo_rating_turf, rt.elo_rating_dirt,
        rt.elo_turf_1200, rt.elo_turf_1600, rt.elo_turf_2000, rt.elo_turf_2400, rt.elo_turf_3000,
        rt.elo_dirt_1200, rt.elo_dirt_1600, rt.elo_dirt_2000, rt.elo_dirt_2400, rt.elo_dirt_3000
    FROM target_data t
    LEFT JOIN ucv_rolling ur ON t.race_id = ur.race_id AND t.horse_id = ur.horse_id
    LEFT JOIN jockey_rolling jr ON t.race_id = jr.race_id AND t.jockey_code = jr.jockey_code
    LEFT JOIN feature_dash_score ds ON t.race_id = ds.race_id AND t.horse_id = ds.horse_id
    LEFT JOIN feature_pace_prediction pp ON t.race_id = pp.race_id
    LEFT JOIN feature_opponent_rating rt ON t.race_id = rt.race_id AND t.horse_id = rt.horse_id
    ORDER BY t.race_date ASC
    """
    
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn)
        log(f"[1/6] メインSQL完了: {len(df)}件取得 ({time.time() - start_time:.2f}秒)")
        
        log("[2/6] 調教データの取得開始...")
        start_time = time.time()
        slope_df = pd.read_sql_query("SELECT * FROM slope_training", conn)
        woodchip_df = pd.read_sql_query("SELECT * FROM woodchip_training", conn)
        log(f"[2/6] 調教データ完了: 坂路 {len(slope_df)}件, WC {len(woodchip_df)}件 ({time.time() - start_time:.2f}秒)")
    
    df = df.drop_duplicates(subset=['race_id', 'horse_id'], keep='last')
    
    log("[3/6] 既存の前処理（型変換・欠損値補完等）の実行...")
    start_time = time.time()
    
    # --- ここから既存の前処理 ---
    df['is_win'] = df['is_win'].astype(int)
    df['win_odds'] = pd.to_numeric(df['win_odds'], errors='coerce') / 10.0
    df['win_odds'] = df['win_odds'].fillna(1.0)
    df['distance'] = pd.to_numeric(df['distance'], errors='coerce')
    df['horse_age'] = pd.to_numeric(df['horse_age'], errors='coerce')
    df['weight_carried'] = pd.to_numeric(df['weight_carried'], errors='coerce') / 10.0

    ucv_median = df['avg_ucv_score_5'].median()
    for col in ['prev_ucv_score', 'max_ucv_score_5', 'avg_ucv_score_5']:
        df[col] = df[col].astype(float).fillna(ucv_median)

    ucv_3f_median = df['avg_ucv_3f_5'].median()
    for col in ['max_ucv_3f_5', 'avg_ucv_3f_5']:
        df[col] = df[col].astype(float).fillna(ucv_3f_median)

    df['jockey_win_rate_100'] = df['jockey_win_rate_100'].astype(float).fillna(df['jockey_win_rate_100'].median())
    df['jockey_top3_rate_100'] = df['jockey_top3_rate_100'].astype(float).fillna(df['jockey_top3_rate_100'].median())
    df['dash_score_median'] = df['dash_score_median'].astype(float).fillna(0.5)
    df['pace_score_top3'] = df['pace_score_top3'].astype(float).fillna(0.5)
    df['leader_margin'] = df['leader_margin'].astype(float).fillna(0.0)
    
    df['elo_rating_turf'] = df['elo_rating_turf'].astype(float).fillna(1000.0)
    df['elo_rating_dirt'] = df['elo_rating_dirt'].astype(float).fillna(1000.0)

    turf_cols = ['elo_turf_1200', 'elo_turf_1600', 'elo_turf_2000', 'elo_turf_2400', 'elo_turf_3000']
    dirt_cols = ['elo_dirt_1200', 'elo_dirt_1600', 'elo_dirt_2000', 'elo_dirt_2400', 'elo_dirt_3000']
    for col in turf_cols + dirt_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(1000.0)

    track_code_int = pd.to_numeric(df['track_code'], errors='coerce').fillna(0)
    is_turf = track_code_int.between(10, 22).values
    dist_nodes = np.array([1200.0, 1600.0, 2000.0, 2400.0, 3000.0])
    sigma = 300.0
    distances = df['distance'].values[:, np.newaxis] 
    weights = np.exp(-((distances - dist_nodes) ** 2) / (2.0 * sigma * sigma))
    normalized_weights = weights / weights.sum(axis=1, keepdims=True)
    effective_turf_elo = np.sum(df[turf_cols].values * normalized_weights, axis=1)
    effective_dirt_elo = np.sum(df[dirt_cols].values * normalized_weights, axis=1)
    df['target_elo_rating'] = np.where(is_turf, effective_turf_elo, effective_dirt_elo)

    df['ucv_score_rank'] = df.groupby('race_id')['avg_ucv_score_5'].rank(ascending=False, method='min')
    df['prev_ucv_score_rank'] = df.groupby('race_id')['prev_ucv_score'].rank(ascending=False, method='min')
    df['max_ucv_score_5_rank'] = df.groupby('race_id')['max_ucv_score_5'].rank(ascending=False, method='min')
    ucv_mean = df.groupby('race_id')['avg_ucv_score_5'].transform('mean')
    ucv_std = df.groupby('race_id')['avg_ucv_score_5'].transform('std')
    df['ucv_score_z'] = np.where(ucv_std > 1e-6, (df['avg_ucv_score_5'] - ucv_mean) / ucv_std, 0.0)

    df['ucv_3f_rank'] = df.groupby('race_id')['avg_ucv_3f_5'].rank(ascending=False, method='min')
    df['ucv_3f_gap_avg_5'] = df['avg_ucv_score_5'] - df['avg_ucv_3f_5']

    df['elo_rating_rank'] = df.groupby('race_id')['target_elo_rating'].rank(ascending=False, method='min')
    elo_mean = df.groupby('race_id')['target_elo_rating'].transform('mean')
    elo_std = df.groupby('race_id')['target_elo_rating'].transform('std')
    df['elo_rating_z'] = np.where(elo_std > 1e-6, (df['target_elo_rating'] - elo_mean) / elo_std, 0.0)
    df['elo_diff_from_mean'] = df['target_elo_rating'] - elo_mean

    df['ucv_elo_gap'] = df['ucv_score_z'] - df['elo_rating_z']
    df['dash_score_rank'] = df.groupby('race_id')['dash_score_median'].rank(ascending=True, method='min')
    weight_mean = df.groupby('race_id')['weight_carried'].transform('mean')
    df['weight_carried_diff'] = df['weight_carried'] - weight_mean

    df['race_date_dt'] = pd.to_datetime(df['race_date'])
    df['prev_race_date_dt'] = pd.to_datetime(df['prev_race_date'], errors='coerce')
    df['prev_race_date_dt'] = df['prev_race_date_dt'].fillna(df['race_date_dt'] - pd.Timedelta(days=90))
    # 前走からの経過日数（interval_days）を算出
    df['interval_days'] = (df['race_date_dt'] - df['prev_race_date_dt']).dt.days

    # --- 新規特徴量の追加（前走からの距離変化、昇級フラグ、東西所属） ---
    df['prev_distance_num'] = pd.to_numeric(df['prev_distance'], errors='coerce')
    df['prev_distance_num'] = df['prev_distance_num'].fillna(df['distance'])
    df['distance_change'] = df['distance'] - df['prev_distance_num']

    class_mapping = {
        '701': 1, '702': 1, '703': 1,
        '003': 2, '004': 2, '005': 2,
        '007': 3, '009': 3, '010': 3,
        '014': 4, '015': 4, '016': 4,
        '000': 5, '999': 5
    }
    df['curr_class_level'] = df['cond_code_youngest'].astype(str).str.strip().map(class_mapping).fillna(2)
    df['prev_class_level'] = df['prev_cond_code_youngest'].astype(str).str.strip().map(class_mapping)
    df['prev_class_level'] = df['prev_class_level'].fillna(df['curr_class_level'])
    df['is_promoted'] = (df['curr_class_level'] > df['prev_class_level']).astype(int)

    df['affiliation_code'] = df['affiliation_code'].astype(str).str.strip().fillna('0')

    # ---------------------------------------------------------
    # 馬体重および派生特徴量の計算
    # ---------------------------------------------------------
    # 馬体重の数値化 (未計量などの空文字・欠損を考慮)
    df['horse_weight_num'] = pd.to_numeric(df['horse_weight'].astype(str).str.strip(), errors='coerce')
    weight_median = df['horse_weight_num'].median()
    df['horse_weight_num'] = df['horse_weight_num'].fillna(weight_median)

    # 馬体重増減の計算
    # JRA-VAN仕様の符号(1: +, 2: -, 3: 0, 4: 比較なし) または 一般的な(+/-)に対応
    sign_map = {'1': 1.0, '2': -1.0, '3': 0.0, '+': 1.0, '-': -1.0}
    signs = df['weight_change_sign'].astype(str).str.strip().map(sign_map).fillna(0.0)
    changes = pd.to_numeric(df['weight_change'].astype(str).str.strip(), errors='coerce').fillna(0.0)
    df['weight_change_num'] = signs * changes

    # 負担重量比率 (負担重量 / 馬体重)
    # ※ weight_carried は既存の前処理ですでに 10.0 で除算され実数化されている前提
    df['carried_weight_ratio'] = (df['weight_carried'] / df['horse_weight_num']).fillna(0.0)

    # ---------------------------------------------------------
    # アプローチ2: 休養明けの仕上がり判定（休養日数 × 馬体重増減）
    # ---------------------------------------------------------
    # 特徴量1: 休養1日あたりの体重変動率 (ゼロ除算防止のため分母に+1)
    # 意図: 長期休養の+10kgと、中1週の+10kgの価値を正規化して比較する
    df['weight_change_per_day'] = (df['weight_change_num'] / (df['interval_days'] + 1)).fillna(0.0)

    # 特徴量2: 短期疲労による馬体減りフラグ (中2週=21日以内 かつ -4kg以上減少)
    # 意図: 詰まったローテーションでの馬体減を「仕上がり」ではなく「疲労」としてペナルティ化する
    df['is_fatigue_loss'] = ((df['interval_days'] <= 21) & (df['weight_change_num'] <= -4.0)).astype(int)

    # ---------------------------------------------------------
    # Route D: 枠順バイアス（Target Encoding 高解像度版）と騎手乗り替わり
    # ---------------------------------------------------------
    # 1. 前走からの乗り替わり判定
    # 1. 前走からの脚質・先行力に関する特徴量生成
    df['corner_4_order_num'] = pd.to_numeric(df['corner_4_order'].astype(str).str.strip(), errors='coerce').fillna(0)
    df['race_horse_count'] = df.groupby('race_id')['horse_id'].transform('count')
    df['corner_4_ratio'] = df['corner_4_order_num'] / df['race_horse_count']
    # '00' や 0 の無効な通過順は NaN とする
    df.loc[df['corner_4_order_num'] <= 0, 'corner_4_ratio'] = np.nan

    df = df.sort_values(['horse_id', 'race_date_dt', 'race_id'])
    
    # 急坂フラグ（中山06, 中京07, 阪神09）
    course_code_str = df['course_code'].astype(str).str.strip().str.zfill(2)
    df['is_steep_slope'] = course_code_str.isin(['06', '07', '09']).astype(int)

    # 過去の急坂コースにおける実績
    df['steep_slope_top3_raw'] = np.where((df['is_steep_slope'] == 1) & (df['is_top3'] == 1), 1.0, 0.0)
    df['steep_slope_top3_count'] = df.groupby('horse_id')['steep_slope_top3_raw'].transform(
        lambda x: x.expanding().sum().shift().fillna(0.0)
    )
    df['steep_slope_run_count'] = df.groupby('horse_id')['is_steep_slope'].transform(
        lambda x: x.expanding().sum().shift().fillna(0.0)
    )
    df['steep_slope_top3_rate'] = (df['steep_slope_top3_count'] / df['steep_slope_run_count']).fillna(0.0)
    df = df.drop(columns=['steep_slope_top3_raw', 'steep_slope_top3_count', 'steep_slope_run_count'])

    # 前走の4角通過順比率
    df['prev_corner_4_ratio'] = df.groupby('horse_id')['corner_4_ratio'].shift(1)
    df['prev_corner_4_ratio'] = df['prev_corner_4_ratio'].fillna(0.5)

    # 過去5走の4角通過順比率平均値
    df['avg_corner_4_ratio_5'] = df.groupby('horse_id')['corner_4_ratio'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    df['avg_corner_4_ratio_5'] = df['avg_corner_4_ratio_5'].fillna(0.5)

    # 前走の脚質
    df['prev_running_style'] = df.groupby('horse_id')['running_style'].shift(1)
    df['prev_running_style'] = df['prev_running_style'].fillna('0').astype(str).str.strip()
    df.loc[df['prev_running_style'].isin(['', 'None', 'nan']), 'prev_running_style'] = '0'

    # 前走の実際の4角通過順位および真の逃げ馬フラグ（前走逃げ、かつ4角1番手）
    df['prev_corner_4_order_num'] = df.groupby('horse_id')['corner_4_order_num'].shift(1)
    df['is_true_escape_prev'] = ((df['prev_running_style'] == '1') & (df['prev_corner_4_order_num'] == 1.0)).astype(int)

    # 前走からの乗り替わり判定
    df['prev_jockey_code'] = df.groupby('horse_id')['jockey_code'].shift(1)
    df['is_jockey_changed'] = (df['jockey_code'] != df['prev_jockey_code']).astype(int)
    df.loc[df['prev_jockey_code'].isna(), 'is_jockey_changed'] = 0

    # 2. 当日乗り替わり判定
    df['is_sameday_jockey_change'] = df['jockey_code_before'].apply(
        lambda x: 1 if pd.notna(x) and str(x).strip() not in ['', '0', '0000', 'None'] else 0
    )

    # 3. 枠順バイアス (コース × トラック × 距離 × 枠番)
    df = df.sort_values(['race_date_dt', 'race_id'])
    
    def expanding_mean_shifted(s):
        return s.expanding().mean().shift().fillna(0.0)

    # 複合グループキーの作成 (例: '05_11_1600_8' = 東京_芝_1600m_8枠)
    df['bias_group_key'] = (
        df['course_code'].astype(str).str.strip() + '_' + 
        df['track_code'].astype(str).str.strip() + '_' + 
        df['distance'].astype(str).str.strip() + '_' + 
        df['bracket_number'].astype(str).str.strip()
    )

    df['bracket_win_rate_highres'] = df.groupby('bias_group_key')['is_win'].transform(expanding_mean_shifted)
    df['bracket_top3_rate_highres'] = df.groupby('bias_group_key')['is_top3'].transform(expanding_mean_shifted)

    df = df.drop(columns=['bias_group_key', 'prev_jockey_code'])
    # ---------------------------------------------------------

    course = df['course_code'].astype(str).str.strip().str.zfill(2)
    track = df['track_code'].astype(str).str.strip()
    is_dirt = track.str.startswith('2') | (track == '29')
    is_outer = track.isin(['12', '13', '16', '18', '19', '22'])
    is_straight = (track == '10')

    sl = pd.Series(300.0, index=df.index)
    sl = np.where(course == '01', np.where(is_dirt, 264.3, 266.1), sl) 
    sl = np.where(course == '02', np.where(is_dirt, 260.3, 262.1), sl) 
    sl = np.where(course == '03', np.where(is_dirt, 295.7, 292.0), sl) 
    sl = np.where(course == '04', np.where(is_dirt, 353.9, np.where(is_straight, 1000.0, np.where(is_outer, 658.7, 358.7))), sl) 
    sl = np.where(course == '05', np.where(is_dirt, 501.6, 525.9), sl) 
    sl = np.where(course == '06', np.where(is_dirt, 308.0, 310.0), sl) 
    sl = np.where(course == '07', np.where(is_dirt, 410.7, 412.5), sl) 
    sl = np.where(course == '08', np.where(is_dirt, 329.1, np.where(is_outer, 403.7, 328.4)), sl) 
    sl = np.where(course == '09', np.where(is_dirt, 352.7, np.where(is_outer, 473.6, 356.5)), sl) 
    sl = np.where(course == '10', np.where(is_dirt, 291.3, 293.0), sl) 
    df['straight_length'] = sl

    # ---------------------------------------------------------
    # Route E: 新規追加特徴量（コース特性、小回り×先行力）
    # ---------------------------------------------------------
    # 主要4場（東京05, 中山06, 京都08, 阪神09）は0、ローカル6場（札幌01, 函館02, 福島03, 新潟04, 中京07, 小倉10）は1
    local_courses = {'01', '02', '03', '04', '07', '10'}
    df['is_local'] = df['course_code'].astype(str).str.strip().str.zfill(2).isin(local_courses).astype(int)
    df['is_small_turn'] = (df['straight_length'] < 350.0).astype(int)
    df['interaction_small_turn_lead'] = df['is_small_turn'] * (1.0 - df['avg_corner_4_ratio_5'])

    # 短い直線フラグ（直線長 320m 未満）
    df['is_short_straight'] = (df['straight_length'] < 320.0).astype(int)
    # 短い直線×後方脚質（直線が短く、後方にいる馬へのペナルティ指標）
    df['interaction_short_straight_back'] = df['is_short_straight'] * df['avg_corner_4_ratio_5']
    # 急坂×馬体重（急坂でのタフさ・パワー指標）
    df['interaction_steep_slope_weight'] = df['is_steep_slope'] * df['horse_weight_num']

    # レース内でのダッシュ力偏差値（Z値）
    dash_mean = df.groupby('race_id')['dash_score_median'].transform('mean')
    dash_std = df.groupby('race_id')['dash_score_median'].transform('std')
    df['dash_score_z'] = np.where(dash_std > 1e-6, (df['dash_score_median'] - dash_mean) / dash_std, 0.0)

    # レース内でのダッシュ力順位
    df['dash_score_in_race_rank'] = df.groupby('race_id')['dash_score_median'].rank(ascending=False, method='min')

    # レース内最速ダッシュフラグ
    df['is_fastest_dash_in_race'] = (df['dash_score_in_race_rank'] == 1).astype(int)

    # レース内の逃げ馬の合計頭数
    df['escape_horse_count_in_race'] = df.groupby('race_id')['prev_running_style'].transform(lambda x: (x == '1').sum())

    # 先行馬の合計頭数
    df['lead_horse_count_in_race'] = df.groupby('race_id')['prev_running_style'].transform(lambda x: (x == '2').sum())
    # 逃げ＋先行馬の合計頭数
    df['front_active_horse_count_in_race'] = df['escape_horse_count_in_race'] + df['lead_horse_count_in_race']
    # 逃げ＋先行馬の割合
    df['front_active_horse_ratio'] = (df['front_active_horse_count_in_race'] / df['race_horse_count']).fillna(0.0)

    # 展開相互作用：逃げ馬頭数×先行力（先行激突ペナルティ）
    df['interaction_escape_conflict_lead'] = df['escape_horse_count_in_race'] * (1.0 - df['avg_corner_4_ratio_5'])

    # 展開相互作用：逃げ馬頭数×後方脚質（前崩れ台頭ボーナス）
    df['interaction_escape_conflict_back'] = df['escape_horse_count_in_race'] * df['avg_corner_4_ratio_5']

    # ---------------------------------------------------------
    # アプローチB改: 競馬場特有・展開および脚質相互作用特徴量
    # ---------------------------------------------------------
    track_code_int_temp = pd.to_numeric(df['track_code'], errors='coerce').fillna(0)
    is_turf_temp = track_code_int_temp.between(10, 22)
    is_dirt_temp = track_code_int_temp.between(23, 29)

    # 小倉ダートフラグ (course_code == '10' かつ ダート)
    df['is_kokura_dirt'] = ((course_code_str == '10') & is_dirt_temp).astype(int)
    
    # 小倉ダート × 真の逃げ馬
    df['interaction_kokura_dirt_true_escape'] = df['is_kokura_dirt'] * df['is_true_escape_prev']
    
    # 小倉ダート × 先行激突ペナルティ
    df['interaction_kokura_dirt_escape_conflict'] = df['is_kokura_dirt'] * df['interaction_escape_conflict_lead']
    
    # 小倉ダート × 前崩れ差し馬ボーナス
    df['interaction_kokura_dirt_escape_conflict_back'] = df['is_kokura_dirt'] * df['interaction_escape_conflict_back']
    
    # 中山芝フラグ (course_code == '06' かつ 芝)
    df['is_nakayama_turf'] = ((course_code_str == '06') & is_turf_temp).astype(int)
    
    # 中山芝 × 先行機動力ボーナス
    df['interaction_nakayama_turf_lead'] = df['is_nakayama_turf'] * (1.0 - df['avg_corner_4_ratio_5'])
    
    # 中山芝 × 差し馬ペナルティ
    df['interaction_nakayama_turf_back'] = df['is_nakayama_turf'] * df['avg_corner_4_ratio_5']

    # ---------------------------------------------------------
    # 新規追加特徴量: 直線長と脚質の交互作用特徴量
    # ---------------------------------------------------------
    df['interaction_straight_length_lead'] = df['straight_length'] * (1.0 - df['avg_corner_4_ratio_5'])
    df['interaction_straight_length_back'] = df['straight_length'] * df['avg_corner_4_ratio_5']

    # 中京ダートフラグ (course_code == '07' かつ ダート)
    df['is_chukyo_dirt'] = ((course_code_str == '07') & is_dirt_temp).astype(int)
    df['interaction_chukyo_dirt_lead'] = df['is_chukyo_dirt'] * (1.0 - df['avg_corner_4_ratio_5'])
    df['interaction_chukyo_dirt_back'] = df['is_chukyo_dirt'] * df['avg_corner_4_ratio_5']

    # 小倉ダート × 先行・差し交互作用
    df['interaction_kokura_dirt_lead'] = df['is_kokura_dirt'] * (1.0 - df['avg_corner_4_ratio_5'])
    df['interaction_kokura_dirt_back'] = df['is_kokura_dirt'] * df['avg_corner_4_ratio_5']

    # 福島ダート1150mフラグ (course_code == '03' かつ ダート かつ 距離1150m)
    df['is_fukushima_dirt_1150'] = ((course_code_str == '03') & is_dirt_temp & (df['distance'] == 1150)).astype(int)
    df['interaction_fukushima_dirt_1150_lead'] = df['is_fukushima_dirt_1150'] * (1.0 - df['avg_corner_4_ratio_5'])
    df['interaction_fukushima_dirt_1150_back'] = df['is_fukushima_dirt_1150'] * df['avg_corner_4_ratio_5']

    cat_cols = ['course_code', 'track_code', 'turf_condition_code', 'dirt_condition_code', 
                'bracket_number', 'horse_number', 'sex_code', 'prev_running_style',
                'weather_code', 'grade_code', 'race_type_code', 'weight_type_code', 'cond_code_youngest',
                'affiliation_code']
    for col in cat_cols:
        if isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].astype('object')
        df[col] = df[col].fillna('0').astype(str).str.strip()
        df[col] = df[col].replace(['nan', 'None', ''], '0')
        df[col] = df[col].astype('category')
    # --- ここまで既存の前処理 ---
    
    log(f"[3/6] 前処理完了 ({time.time() - start_time:.2f}秒)")

    # ---------------------------------------------------------
    # 調教データ結合関数 (DuckDBによる直積回避版)
    # ---------------------------------------------------------
    def process_training_data(tdf, prefix, sort_col, feature_cols):
        if tdf.empty:
            return pd.DataFrame()
            
        tdf['training_date_dt'] = pd.to_datetime(tdf['training_date'], format='%Y%m%d', errors='coerce')
        
        cols_to_extract = list(set([sort_col] + feature_cols))
        for col in cols_to_extract:
            tdf[col + '_sec'] = pd.to_numeric(tdf[col].replace('0000', np.nan), errors='coerce') / 10.0
            
        # --- 追加特徴量の算出開始 ---
        tdf = tdf.sort_values(['blood_reg_number', 'training_date_dt'])
        
        # 1. 前日までの自己ベストタイム (データリーク防止のためgroupbyしてtransformとshift(1)を使用)
        tdf['prev_best_time'] = tdf.groupby('blood_reg_number')[sort_col + '_sec'].transform(
            lambda x: x.cummin().shift(1)
        )
        # 自己ベスト比率
        tdf['best_time_ratio'] = (tdf[sort_col + '_sec'] / tdf['prev_best_time']).fillna(1.0)
        
        # 2. ラスト1Fの加速ラップ（終い重点）
        if prefix == 'slope':
            # 坂路: ラスト2F目(lap_time_2f_1f_sec) と ラスト1F(lap_time_1f_0m_sec) の比較
            tdf['lap_diff'] = tdf['lap_time_2f_1f_sec'] - tdf['lap_time_1f_0m_sec']
            tdf['is_acceleration'] = (tdf['lap_diff'] > 0.0).astype(float)
        elif prefix == 'woodchip':
            # ウッドチップ: ラスト2Fタイム(total_time_2f_sec) と ラスト1F(lap_time_1f_0m_sec) の比較
            # ラスト2F目のハロンタイム ＝ total_time_2f_sec - lap_time_1f_0m_sec
            tdf['lap_diff'] = (tdf['total_time_2f_sec'] - tdf['lap_time_1f_0m_sec']) - tdf['lap_time_1f_0m_sec']
            tdf['is_acceleration'] = (tdf['lap_diff'] > 0.0).astype(float)
        else:
            tdf['lap_diff'] = 0.0
            tdf['is_acceleration'] = 0.0
            
        tdf['lap_diff'] = tdf['lap_diff'].fillna(0.0)
        tdf['is_acceleration'] = tdf['is_acceleration'].fillna(0.0)
        # --- 追加特徴量の算出終了 ---
            
        sec_cols = [c + '_sec' for c in cols_to_extract]
        new_feature_cols = ['best_time_ratio', 'lap_diff', 'is_acceleration']
        
        df_dates = df[['race_id', 'horse_id', 'race_date_dt', 'prev_race_date_dt']]
        tdf_target = tdf[['blood_reg_number', 'training_date_dt'] + sec_cols + new_feature_cols]
        
        log(f"  - {prefix} 結合前: メイン {len(df_dates)}件 / 調教 {len(tdf_target)}件")
        
        # --- DuckDBによる非等価結合 (Range Join) ---
        all_select_cols = [f't."{c}"' for c in (sec_cols + new_feature_cols)]
        sec_cols_str = ", ".join(all_select_cols)
        query = f"""
            SELECT 
                r.race_id, 
                r.horse_id, 
                r.race_date_dt, 
                t.training_date_dt,
                {sec_cols_str}
            FROM df_dates AS r
            INNER JOIN tdf_target AS t
                ON r.horse_id = t.blood_reg_number
                AND t.training_date_dt > r.prev_race_date_dt
                AND t.training_date_dt <= r.race_date_dt
        """
        
        # クエリを実行し、結果をPandas DataFrameとして受け取る
        valid_train = duckdb.query(query).df()
        log(f"  - {prefix} 結合完了: DuckDB抽出により {len(valid_train)}行 に最適化されました")
        
        counts = valid_train.groupby(['race_id', 'horse_id']).size().rename(f'{prefix}_count')
        
        if valid_train.empty:
            return pd.DataFrame(counts).reset_index()

        two_weeks = valid_train[(valid_train['race_date_dt'] - valid_train['training_date_dt']).dt.days <= 14]
        sorted_train = two_weeks.dropna(subset=[sort_col + '_sec']).sort_values(['race_id', 'horse_id', sort_col + '_sec'], ascending=[True, True, True])
        fastest = sorted_train.groupby(['race_id', 'horse_id']).head(2).copy()
        
        if fastest.empty:
            return pd.DataFrame(counts).reset_index()

        fastest['rank'] = fastest.groupby(['race_id', 'horse_id']).cumcount() + 1
        value_cols = [c + '_sec' for c in feature_cols] + new_feature_cols
        pivoted = fastest.pivot(index=['race_id', 'horse_id'], columns='rank', values=value_cols)
        pivoted.columns = [f"{prefix}_{col.replace('_sec', '')}_{rank}" for col, rank in pivoted.columns]
        
        return pd.concat([counts, pivoted], axis=1).reset_index()
    
    log("[4/6] 坂路調教データの特徴量抽出...")
    start_time = time.time()
    feature_cols_slope = ['total_time_4f', 'lap_time_2f_1f', 'lap_time_1f_0m']
    slope_features = process_training_data(slope_df, 'slope', 'total_time_4f', feature_cols_slope)
    log(f"[4/6] 坂路データ抽出完了 ({time.time() - start_time:.2f}秒)")

    log("[5/6] ウッドチップ調教データの特徴量抽出...")
    start_time = time.time()
    feature_cols_wc = ['total_time_5f', 'total_time_4f', 'total_time_2f', 'lap_time_1f_0m']
    woodchip_features = process_training_data(woodchip_df, 'woodchip', 'total_time_4f', feature_cols_wc)
    log(f"[5/6] ウッドチップデータ抽出完了 ({time.time() - start_time:.2f}秒)")

    log("[6/6] 特徴量の最終マージと欠損値補完...")
    start_time = time.time()
    if not slope_features.empty:
        df = df.merge(slope_features, on=['race_id', 'horse_id'], how='left')
    if not woodchip_features.empty:
        df = df.merge(woodchip_features, on=['race_id', 'horse_id'], how='left')
    # ---------------------------------------------------------
    # 追加: 調教密度（回数 ÷ 休養日数）の計算
    # ---------------------------------------------------------
    # 欠損値(NaN)は0回として処理し、休養日数+1で割る
    df['slope_per_day'] = (df['slope_count'].fillna(0) / (df['interval_days'] + 1)).fillna(0.0)
    df['woodchip_per_day'] = (df['woodchip_count'].fillna(0) / (df['interval_days'] + 1)).fillna(0.0)
    # ---------------------------------------------------------

    for prefix, f_cols in [('slope', feature_cols_slope), ('woodchip', feature_cols_wc)]:
        if f'{prefix}_count' in df.columns:
            df[f'{prefix}_count'] = df[f'{prefix}_count'].fillna(0)
            
        # タイム系の補完 (99.9秒)
        for col in f_cols:
            for rank in [1, 2]:
                cname = f'{prefix}_{col}_{rank}'
                if cname in df.columns:
                    df[cname] = df[cname].fillna(99.9)
                    
        # 新調教特徴量の補完
        for rank in [1, 2]:
            cname = f'{prefix}_best_time_ratio_{rank}'
            if cname in df.columns:
                df[cname] = df[cname].fillna(1.0)
                
            cname = f'{prefix}_lap_diff_{rank}'
            if cname in df.columns:
                df[cname] = df[cname].fillna(0.0)
                
            cname = f'{prefix}_is_acceleration_{rank}'
            if cname in df.columns:
                df[cname] = df[cname].fillna(0.0)

    # ---------------------------------------------------------
    # ペース予測モデルによる pred_lap_diff の追加
    # ---------------------------------------------------------
    log("ペース予測モデルの推論中...")
    try:
        import os
        import lightgbm as lgb
        pace_model_path = os.path.join(os.path.dirname(__file__), 'lgbm_pace_model.txt') if '__file__' in locals() else 'lgbm_pace_model.txt'
        if not os.path.exists(pace_model_path):
            pace_model_path = 'lgbm_pace_model.txt'
            
        if os.path.exists(pace_model_path):
            booster_pace = lgb.Booster(model_file=pace_model_path)
            
            # 必要なカラムの準備
            pace_df = df[[
                'race_id', 'distance', 'course_code', 'track_code', 'turf_condition_code', 'dirt_condition_code', 
                'race_horse_count', 'pace_score_top3', 'leader_margin',
                'straight_length', 'is_steep_slope', 'is_local', 
                'escape_horse_count_in_race', 'lead_horse_count_in_race', 
                'front_active_horse_count_in_race', 'front_active_horse_ratio',
                'weather_code', 'grade_code', 'race_type_code', 'weight_type_code', 'cond_code_youngest'
            ]].drop_duplicates(subset=['race_id']).copy()
            pace_df = pace_df.rename(columns={'race_horse_count': 'started_count'})
            
            # 型変換
            pace_df['distance'] = pd.to_numeric(pace_df['distance'], errors='coerce').fillna(1600).astype(int)
            pace_df['started_count'] = pd.to_numeric(pace_df['started_count'], errors='coerce').fillna(10).astype(int)
            pace_df['pace_score_top3'] = pace_df['pace_score_top3'].astype(float).fillna(0.5)
            pace_df['leader_margin'] = pace_df['leader_margin'].astype(float).fillna(0.0)
            pace_df['straight_length'] = pace_df['straight_length'].astype(float).fillna(300.0)
            pace_df['is_steep_slope'] = pace_df['is_steep_slope'].astype(int).fillna(0)
            pace_df['is_local'] = pace_df['is_local'].astype(int).fillna(0)
            pace_df['escape_horse_count_in_race'] = pace_df['escape_horse_count_in_race'].astype(int).fillna(0)
            pace_df['lead_horse_count_in_race'] = pace_df['lead_horse_count_in_race'].astype(int).fillna(0)
            pace_df['front_active_horse_count_in_race'] = pace_df['front_active_horse_count_in_race'].astype(int).fillna(0)
            pace_df['front_active_horse_ratio'] = pace_df['front_active_horse_ratio'].astype(float).fillna(0.0)
            
            cat_cols = ['course_code', 'track_code', 'turf_condition_code', 'dirt_condition_code',
                        'weather_code', 'grade_code', 'race_type_code', 'weight_type_code', 'cond_code_youngest']
            for col in cat_cols:
                pace_df[col] = pace_df[col].astype(str).str.strip().fillna('-1').astype('category')
                
            pace_features = [
                'distance', 'course_code', 'track_code', 
                'turf_condition_code', 'dirt_condition_code', 
                'started_count', 'pace_score_top3', 'leader_margin',
                'straight_length', 'is_steep_slope', 'is_local',
                'escape_horse_count_in_race', 'lead_horse_count_in_race', 
                'front_active_horse_count_in_race', 'front_active_horse_ratio',
                'weather_code', 'grade_code', 'race_type_code', 'weight_type_code', 'cond_code_youngest'
            ]
            
            # 推論
            pace_df['pred_lap_diff'] = booster_pace.predict(pace_df[pace_features])
            
            # メインのdfにマージ
            df = df.merge(pace_df[['race_id', 'pred_lap_diff']], on='race_id', how='left')
            df['pred_lap_diff'] = df['pred_lap_diff'].fillna(0.0)
            log("ペース予測モデルの推論完了")
        else:
            log("警告: ペース予測モデルファイルが見つかりません。pred_lap_diff を 0.0 で初期化します。")
            df['pred_lap_diff'] = 0.0
    except Exception as e:
        log(f"警告: ペース予測モデルの推論中にエラーが発生しました: {e}")
        df['pred_lap_diff'] = 0.0

    df = df.drop(columns=['race_date_dt', 'prev_race_date_dt'], errors='ignore')
    log(f"[6/6] 全処理完了 ({time.time() - start_time:.2f}秒)")

    return df

if __name__ == "__main__":
    db_path = 'C:/Ugaura/sqlite/jra_race.db'
    load_feature_matrix(db_path)
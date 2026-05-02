import sqlite3
import pandas as pd
import numpy as np
from scipy.stats import trim_mean

# データベースの絶対パス
DB_PATH = r"C:\sqlite\jra_race.db"

# JRAの中央10場のコードリスト
JRA_COURSE_CODES = ('01', '02', '03', '04', '05', '06', '07', '08', '09', '10')

def parse_running_time(t_str):
    """JRA-VANのタイム文字列(例: '1234')を秒数(83.4)に変換"""
    if pd.isna(t_str) or not str(t_str).strip().isdigit():
        return None
    t = str(t_str).strip().zfill(4)
    m = int(t[0])
    s = int(t[1:3])
    ms = int(t[3])
    return m * 60 + s + ms * 0.1

def parse_3f_time(t_str):
    """JRA-VANの上がり3F文字列(例: '345')を秒数(34.5)に変換"""
    if pd.isna(t_str) or not str(t_str).strip().isdigit():
        return 0.0 # 欠損・異常値は0.0にして後で除外する
    t = str(t_str).strip().zfill(3)
    s = int(t[0:2])
    ms = int(t[2])
    return s + ms * 0.1

def get_track_category(code):
    """良馬場判定のためだけに大分類(芝/ダート/障害)を返す"""
    c = str(code).strip()
    if c in [str(i) for i in range(10, 23)]: return '芝'
    if c in [str(i) for i in range(23, 30)]: return 'ダート'
    if c in [str(i) for i in range(51, 60)]: return '障害'
    return 'その他'

def generate_base_time_master(df_horse_race: pd.DataFrame) -> pd.DataFrame:
    # 降級制度廃止後(2019-06-01以降) かつ 良馬場 かつ タイムが存在するデータ
    df_target = df_horse_race[
        (df_horse_race['race_date'] >= '2019-06-01') & 
        (df_horse_race['is_good_condition'] == True)
    ]
    group_keys = ['course_code', 'track_code', 'distance', 'race_type_code', 'cond_code_youngest']
    
    def calculate_stats(group):
        times = group['time_seconds'].values
        count = len(times)
        if count == 0:
            return pd.Series({'base_time_seconds': np.nan, 'std_dev': np.nan, 'sample_count': 0})
        elif count < 5:
            # サンプル不足時は単純平均
            return pd.Series({'base_time_seconds': np.mean(times), 'std_dev': np.std(times, ddof=1) if count > 1 else 0.0, 'sample_count': count})
        else:
            # トリム平均（上下10%の外れ値を除外）
            t_mean = trim_mean(times, 0.1)
            lower_bound, upper_bound = np.percentile(times, 10), np.percentile(times, 90)
            trimmed_times = times[(times >= lower_bound) & (times <= upper_bound)]
            t_std = np.std(trimmed_times, ddof=1) if len(trimmed_times) > 1 else 0.0
            return pd.Series({'base_time_seconds': t_mean, 'std_dev': t_std, 'sample_count': count})

    df_base_time = df_target.groupby(group_keys).apply(calculate_stats, include_groups=False).reset_index()
    return df_base_time.dropna(subset=['base_time_seconds'])

def calculate_ucv_3f_base_time(df_horse_race: pd.DataFrame) -> pd.DataFrame:
    """上がり3ハロンの基準タイムと標準偏差を算出する"""
    df_valid = df_horse_race[df_horse_race['last_3f_time_seconds'] > 0].copy()

    # 3F基準タイムは、直線の長さが異なるため内外回りを区別する (track_codeを使用)
    group_keys = [
        'course_code',
        'track_code', 
        'distance',
        'race_type_code',
        'cond_code_youngest'
    ]

    df_base_3f = df_valid.groupby(group_keys)['last_3f_time_seconds'].agg(
        base_3f_time_seconds='median',
        base_3f_std='std',
        sample_count='count'
    ).reset_index()

    # 統計的補正処理 (0除算やNaN回避)
    default_std = 0.6
    df_base_3f['base_3f_std'] = df_base_3f['base_3f_std'].fillna(default_std)
    df_base_3f.loc[df_base_3f['base_3f_std'] < 0.1, 'base_3f_std'] = default_std

    return df_base_3f

def calculate_track_bias(df_horse_race: pd.DataFrame, df_base_time: pd.DataFrame) -> pd.DataFrame:
    # レースごとの中央値タイムを算出
    df_race_median = df_horse_race.groupby('race_id')['time_seconds'].median().reset_index(name='race_median_time')
    
    race_meta_cols = ['race_id', 'race_date', 'course_code', 'track_code', 'track_category_for_cond', 'distance', 'race_type_code', 'cond_code_youngest', 'race_number']
    df_race_info = df_horse_race[race_meta_cols].drop_duplicates()
    
    df_race = pd.merge(df_race_info, df_race_median, on='race_id')
    
    # 基準タイムと結合して偏差を算出
    group_keys = ['course_code', 'track_code', 'distance', 'race_type_code', 'cond_code_youngest']
    df_race = pd.merge(df_race, df_base_time[group_keys + ['base_time_seconds']], on=group_keys, how='left')
    df_race['deviation'] = df_race['race_median_time'] - df_race['base_time_seconds']
    
    df_race = df_race.dropna(subset=['deviation']).sort_values(['race_date', 'course_code', 'track_category_for_cond', 'race_number'])
    
    def compute_smoothed_bias(group):
        # 移動中央値と縮小推定の適用
        group['rolling_deviation'] = group['deviation'].rolling(window=3, min_periods=1, center=True).median()
        N = len(group)
        alpha = 3.0
        group['track_bias_seconds'] = (N * group['rolling_deviation'] + alpha * 0) / (N + alpha)
        return group

    # グループ化のキーを track_category_for_cond に変更（内・外回りを同一視してN数を稼ぐ）
    df_bias = df_race.groupby(['race_date', 'course_code', 'track_category_for_cond']).apply(compute_smoothed_bias, include_groups=False).reset_index()
    
    return df_bias[['race_id', 'race_date', 'course_code', 'track_code', 'track_bias_seconds']]

def calculate_track_bias_3f(df_horse_race: pd.DataFrame, df_base_time_3f: pd.DataFrame) -> pd.DataFrame:
    """上がり3ハロン専用の馬場差を算出する"""
    df_valid = df_horse_race[df_horse_race['last_3f_time_seconds'] > 0].copy()
    df_race_median = df_valid.groupby('race_id')['last_3f_time_seconds'].median().reset_index(name='race_median_3f')
    
    race_meta_cols = ['race_id', 'race_date', 'course_code', 'track_code', 'track_category_for_cond', 'distance', 'race_type_code', 'cond_code_youngest', 'race_number']
    df_race_info = df_horse_race[race_meta_cols].drop_duplicates()
    df_race = pd.merge(df_race_info, df_race_median, on='race_id')
    
    group_keys = ['course_code', 'track_code', 'distance', 'race_type_code', 'cond_code_youngest']
    df_race = pd.merge(df_race, df_base_time_3f[group_keys + ['base_3f_time_seconds']], on=group_keys, how='left')
    
    df_race['deviation_3f'] = df_race['race_median_3f'] - df_race['base_3f_time_seconds']
    df_race = df_race.dropna(subset=['deviation_3f']).sort_values(['race_date', 'course_code', 'track_category_for_cond', 'race_number'])
    
    def compute_smoothed_bias(group):
        group['rolling_dev_3f'] = group['deviation_3f'].rolling(window=3, min_periods=1, center=True).median()
        N = len(group)
        alpha = 3.0
        group['track_bias_3f_seconds'] = (N * group['rolling_dev_3f'] + alpha * 0) / (N + alpha)
        return group

    df_bias_3f = df_race.groupby(['race_date', 'course_code', 'track_category_for_cond']).apply(compute_smoothed_bias, include_groups=False).reset_index()
    return df_bias_3f[['race_id', 'race_date', 'course_code', 'track_code', 'track_bias_3f_seconds']]

def calculate_ucv(df_horse_race: pd.DataFrame, df_base_time: pd.DataFrame, df_track_bias: pd.DataFrame) -> pd.DataFrame:
    group_keys = ['course_code', 'track_code', 'distance', 'race_type_code', 'cond_code_youngest']
    df_ucv = pd.merge(df_horse_race, df_base_time, on=group_keys, how='left')
    df_ucv = pd.merge(df_ucv, df_track_bias[['race_id', 'track_bias_seconds']], on='race_id', how='left')
    df_ucv['track_bias_seconds'] = df_ucv['track_bias_seconds'].fillna(0.0)
    df_ucv['corrected_time'] = df_ucv['time_seconds'] - df_ucv['track_bias_seconds']
    
    def compute_z_score(row):
        if pd.isna(row['base_time_seconds']) or pd.isna(row['corrected_time']): return None
        if pd.isna(row['std_dev']) or row['std_dev'] <= 0: return 0.0
        return (row['base_time_seconds'] - row['corrected_time']) / row['std_dev']

    df_ucv['ucv_score'] = df_ucv.apply(compute_z_score, axis=1)
    return df_ucv[['race_id', 'horse_number', 'time_seconds', 'track_bias_seconds', 'base_time_seconds', 'corrected_time', 'ucv_score']]

def calculate_ucv_3f(df_horse_race: pd.DataFrame, df_base_time_3f: pd.DataFrame, df_track_bias_3f: pd.DataFrame) -> pd.DataFrame:
    """上がり3FのUCVスコア(Zスコア)を算出する"""
    group_keys = ['course_code', 'track_code', 'distance', 'race_type_code', 'cond_code_youngest']
    df_ucv_3f = pd.merge(df_horse_race, df_base_time_3f, on=group_keys, how='left')
    df_ucv_3f = pd.merge(df_ucv_3f, df_track_bias_3f[['race_id', 'track_bias_3f_seconds']], on='race_id', how='left')
    
    df_ucv_3f['track_bias_3f_seconds'] = df_ucv_3f['track_bias_3f_seconds'].fillna(0.0)
    df_ucv_3f['corrected_3f_time'] = df_ucv_3f['last_3f_time_seconds'] - df_ucv_3f['track_bias_3f_seconds']
    
    def compute_3f_z_score(row):
        if pd.isna(row['base_3f_time_seconds']) or pd.isna(row['corrected_3f_time']) or row['last_3f_time_seconds'] <= 0:
            return None
        if pd.isna(row['base_3f_std']) or row['base_3f_std'] <= 0:
            return 0.0
        return (row['base_3f_time_seconds'] - row['corrected_3f_time']) / row['base_3f_std']

    df_ucv_3f['ucv_3f_score'] = df_ucv_3f.apply(compute_3f_z_score, axis=1)
    return df_ucv_3f[['race_id', 'horse_number', 'last_3f_time_seconds', 'track_bias_3f_seconds', 'base_3f_time_seconds', 'corrected_3f_time', 'ucv_3f_score']]

def save_to_database(conn: sqlite3.Connection, df_base_time: pd.DataFrame, df_track_bias: pd.DataFrame, df_ucv: pd.DataFrame, 
                     df_base_time_3f: pd.DataFrame, df_track_bias_3f: pd.DataFrame, df_ucv_3f: pd.DataFrame):
    """計算結果をデータベースに保存する（上がり3Fデータ対応版）"""
    cursor = conn.cursor()
    print("   -> 既存の計算データをクリアしています...")
    cursor.execute("DELETE FROM ucv_base_time")
    cursor.execute("DELETE FROM ucv_track_bias")
    cursor.execute("DELETE FROM horse_race_ucv")
    cursor.execute("DELETE FROM ucv_3f_base_time")
    cursor.execute("DELETE FROM ucv_3f_track_bias")
    cursor.execute("DELETE FROM horse_race_ucv_3f")
    conn.commit()
    
    current_jst = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    
    df_base_time['last_updated'] = current_jst
    df_track_bias['last_updated'] = current_jst
    df_base_time_3f['last_updated'] = current_jst
    df_track_bias_3f['last_updated'] = current_jst

    df_base_time.to_sql('ucv_base_time', conn, if_exists='append', index=False, method='multi', chunksize=500)
    df_track_bias.to_sql('ucv_track_bias', conn, if_exists='append', index=False, method='multi', chunksize=500)
    df_base_time_3f.to_sql('ucv_3f_base_time', conn, if_exists='append', index=False, method='multi', chunksize=500)
    df_track_bias_3f.to_sql('ucv_3f_track_bias', conn, if_exists='append', index=False, method='multi', chunksize=500)
    
    # --- 通常UCV ---
    df_ucv = df_ucv.drop_duplicates(subset=['race_id', 'horse_number'], keep='last').replace({np.nan: None})
    data_tuples_ucv = list(df_ucv.itertuples(index=False, name=None))
    sql_ucv = """
    INSERT OR REPLACE INTO horse_race_ucv 
    (race_id, horse_number, time_seconds, applied_track_bias, applied_base_time, corrected_time, ucv_score)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    cursor.executemany(sql_ucv, data_tuples_ucv)

    # --- 上がり3F UCV ---
    df_ucv_3f = df_ucv_3f.drop_duplicates(subset=['race_id', 'horse_number'], keep='last').replace({np.nan: None})
    data_tuples_ucv_3f = list(df_ucv_3f.itertuples(index=False, name=None))
    sql_ucv_3f = """
    INSERT OR REPLACE INTO horse_race_ucv_3f 
    (race_id, horse_number, last_3f_time_seconds, applied_track_bias_3f, applied_base_time_3f, corrected_3f_time, ucv_3f_score)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    cursor.executemany(sql_ucv_3f, data_tuples_ucv_3f)
    conn.commit()

def main():
    print("1. データベースから生データを抽出中 (JRA 10場限定)...")
    conn = sqlite3.connect(DB_PATH)
    
    query_race = f"""
        SELECT 
            year || month_day || course_code || times || day || race_number AS race_id,
            year || '-' || substr(month_day, 1, 2) || '-' || substr(month_day, 3, 2) AS race_date,
            course_code, 
            track_code, 
            CAST(distance AS INTEGER) AS distance, 
            turf_condition_code, 
            dirt_condition_code, 
            race_type_code, 
            cond_code_youngest, 
            race_number 
        FROM race_detail
        WHERE course_code IN {JRA_COURSE_CODES}
    """
    
    query_horse = f"""
        SELECT 
            year || month_day || course_code || times || day || race_number AS race_id,
            horse_number, 
            running_time,
            lap_time_back_3f 
        FROM horse_race_info
        WHERE course_code IN {JRA_COURSE_CODES}
    """
    
    df_race_info = pd.read_sql_query(query_race, conn)
    df_horse_info = pd.read_sql_query(query_horse, conn)
    
    df_horse_race = pd.merge(df_horse_info, df_race_info, on='race_id')
    
    print("2. データのクレンジングとフォーマット変換を実行中...")
    df_horse_race['time_seconds'] = df_horse_race['running_time'].apply(parse_running_time)
    df_horse_race['last_3f_time_seconds'] = df_horse_race['lap_time_back_3f'].apply(parse_3f_time)
    
    df_horse_race = df_horse_race[df_horse_race['time_seconds'] >= 50.0]
    df_horse_race['track_category_for_cond'] = df_horse_race['track_code'].apply(get_track_category)
    df_horse_race = df_horse_race[df_horse_race['track_category_for_cond'] != '障害']
    
    df_horse_race['is_good_condition'] = False
    df_horse_race.loc[df_horse_race['track_category_for_cond'] == '芝', 'is_good_condition'] = (df_horse_race['turf_condition_code'] == '1')
    df_horse_race.loc[df_horse_race['track_category_for_cond'] == 'ダート', 'is_good_condition'] = (df_horse_race['dirt_condition_code'] == '1')

    print("3. 基準タイムマスタ(走破タイム・上がり3F)を算出中...")
    df_base_time = generate_base_time_master(df_horse_race)
    df_base_time_3f = calculate_ucv_3f_base_time(df_horse_race)
    
    print("4. 当日の馬場差・上がり馬場差を算出中...")
    df_track_bias = calculate_track_bias(df_horse_race, df_base_time)
    df_track_bias_3f = calculate_track_bias_3f(df_horse_race, df_base_time_3f)
    
    print("5. 全出走馬のUCVおよびUCV_3Fを算出中...")
    df_ucv = calculate_ucv(df_horse_race, df_base_time, df_track_bias)
    df_ucv_3f = calculate_ucv_3f(df_horse_race, df_base_time_3f, df_track_bias_3f)
    
    print("6. データベースへ計算結果を保存中...")
    save_to_database(conn, df_base_time, df_track_bias, df_ucv, df_base_time_3f, df_track_bias_3f, df_ucv_3f)
    
    conn.close()
    print("全ての処理が完了しました。")

if __name__ == "__main__":
    main()
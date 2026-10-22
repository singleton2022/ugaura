import sqlite3
import pandas as pd
import numpy as np
from scipy.stats import trim_mean

# データベースの絶対パス
DB_PATH = r"C:\Ugaura\sqlite\jra_race.db"

# JRAの中央10場のコードリスト
JRA_COURSE_CODES = ('01', '02', '03', '04', '05', '06', '07', '08', '09', '10')

def parse_running_time(t_str):
    if pd.isna(t_str) or not str(t_str).strip().isdigit():
        return None
    t = str(t_str).strip().zfill(4)
    if t == '0000':
        return None
    m = int(t[0])
    s = int(t[1:3])
    ms = int(t[3])
    return m * 60 + s + ms * 0.1

def parse_3f_time(t_str):
    if pd.isna(t_str) or not str(t_str).strip().isdigit():
        return 0.0
    t = str(t_str).strip().zfill(3)
    s = int(t[0:2])
    ms = int(t[2])
    return s + ms * 0.1

def get_track_category(code):
    c = str(code).strip()
    if c in [str(i) for i in range(10, 23)]: return '芝'
    if c in [str(i) for i in range(23, 30)]: return 'ダート'
    if c in [str(i) for i in range(51, 60)]: return '障害'
    return 'その他'

# 全国標準クラス期待偏差率（1勝クラス水準 = 0.000）
CLASS_RATE_TIME = {
    '芝': {
        '0_新馬未勝利': 0.0140,
        '1_1勝クラス': 0.0000,
        '2_2勝クラス': -0.0037,
        '3_3勝クラス': -0.0087,
        '4_OP重賞': -0.0088
    },
    'ダート': {
        '0_新馬未勝利': 0.0176,
        '1_1勝クラス': 0.0000,
        '2_2勝クラス': -0.0083,
        '3_3勝クラス': -0.0149,
        '4_OP重賞': -0.0223
    }
}

CLASS_RATE_3F = {
    '芝': {
        '0_新馬未勝利': 0.0167,
        '1_1勝クラス': 0.0000,
        '2_2勝クラス': -0.0073,
        '3_3勝クラス': -0.0102,
        '4_OP重賞': -0.0078
    },
    'ダート': {
        '0_新馬未勝利': 0.0243,
        '1_1勝クラス': 0.0000,
        '2_2勝クラス': -0.0110,
        '3_3勝クラス': -0.0174,
        '4_OP重賞': -0.0261
    }
}

def get_class_group(cond):
    c = str(cond).strip() if pd.notna(cond) else ''
    if c in ['701', '702', '703']:
        return '0_新馬未勝利'
    if c == '005':
        return '1_1勝クラス'
    if c == '010':
        return '2_2勝クラス'
    if c == '016':
        return '3_3勝クラス'
    return '4_OP重賞'

def generate_base_time_master(df_horse_race: pd.DataFrame) -> pd.DataFrame:
    """
    コース × トラック × 距離ごとの全クラス統一基準タイムおよび標準偏差を算出。
    出走馬の走破タイムを全国標準クラス（1勝クラス水準）に換算した上でトリム平均（上下10%）を算出し、
    中央とローカルのクラス構成比（未勝利比率の偏り）による甘さ・厳しさを完全に中立化する。
    """
    df_target = df_horse_race[
        (df_horse_race['race_date'] >= '2019-06-01') & 
        (df_horse_race['is_good_condition'] == True)
    ].copy()
    
    def standardize_time(row):
        t_cat = row['track_category_for_cond']
        if t_cat not in CLASS_RATE_TIME:
            return row['time_seconds']
        cg = get_class_group(row.get('cond_code_youngest'))
        rate = CLASS_RATE_TIME[t_cat].get(cg, 0.0)
        return row['time_seconds'] / (1.0 + rate)
    
    df_target['std_time_seconds'] = df_target.apply(standardize_time, axis=1)
    
    rows = []
    for (course, track, dist), group in df_target.groupby(['course_code', 'track_code', 'distance']):
        times = group['std_time_seconds'].values
        count = len(times)
        min_std = max(0.9, float(dist) * 0.0006)
        
        if count == 0:
            continue
        elif count < 5:
            base_t = float(np.mean(times))
            std_val = float(np.std(times, ddof=1)) if count > 1 else min_std
            rows.append({
                'course_code': str(course),
                'track_code': str(track),
                'distance': int(dist),
                'base_time_seconds': base_t,
                'std_dev': max(std_val, min_std),
                'sample_count': count
            })
        else:
            t_mean = float(trim_mean(times, 0.10))
            lower_bound, upper_bound = np.percentile(times, 10), np.percentile(times, 90)
            trimmed_times = times[(times >= lower_bound) & (times <= upper_bound)]
            t_std = float(np.std(trimmed_times, ddof=1)) if len(trimmed_times) > 1 else min_std
            rows.append({
                'course_code': str(course),
                'track_code': str(track),
                'distance': int(dist),
                'base_time_seconds': t_mean,
                'std_dev': max(t_std, min_std),
                'sample_count': count
            })
            
    return pd.DataFrame(rows)

def calculate_ucv_3f_base_time(df_horse_race: pd.DataFrame) -> pd.DataFrame:
    """
    コース × トラック × 距離ごとの全クラス統一上がり3F基準タイムおよび標準偏差を算出。
    上がり3Fタイムを全国標準クラス（1勝クラス水準）に換算した上で中央値を算出する。
    """
    df_valid = df_horse_race[
        (df_horse_race['last_3f_time_seconds'] > 0) &
        (df_horse_race['race_date'] >= '2019-06-01') & 
        (df_horse_race['is_good_condition'] == True)
    ].copy()

    def standardize_3f(row):
        t_cat = row['track_category_for_cond']
        if t_cat not in CLASS_RATE_3F:
            return row['last_3f_time_seconds']
        cg = get_class_group(row.get('cond_code_youngest'))
        rate = CLASS_RATE_3F[t_cat].get(cg, 0.0)
        return row['last_3f_time_seconds'] / (1.0 + rate)

    df_valid['std_3f_time_seconds'] = df_valid.apply(standardize_3f, axis=1)

    group_keys = ['course_code', 'track_code', 'distance']
    df_base_3f = df_valid.groupby(group_keys)['std_3f_time_seconds'].agg(
        base_3f_time_seconds='median',
        base_3f_std='std',
        sample_count='count'
    ).reset_index()

    default_std = 0.8
    df_base_3f['base_3f_std'] = df_base_3f['base_3f_std'].fillna(default_std)
    df_base_3f.loc[df_base_3f['base_3f_std'] < 0.4, 'base_3f_std'] = default_std

    return df_base_3f

def calculate_track_bias(df_horse_race: pd.DataFrame, df_base_time: pd.DataFrame) -> pd.DataFrame:
    """
    各レースの走破タイム中央値と統一基準タイムの乖離から、開催日・競馬場・馬場種別ごとのトラックバイアスを算出。
    - パーセンテージ偏差に換算してトリム平均（20%）を行う。
    - レース固有のクラス差を控除して純粋な馬場バイアス率を抽出。
    - 幼駒特有の展開・能力差が大きい「2歳新馬戦」は馬場差の基準測定から除外。
    - 縮小推定（ベイズ平滑化）後、各コース基準タイムに乗じて補正秒数を導出。
    """
    df_race_median = df_horse_race.groupby('race_id')['time_seconds'].median().reset_index(name='race_median_time')
    
    race_meta_cols = ['race_id', 'race_date', 'course_code', 'track_code', 'track_category_for_cond', 'distance', 'race_number', 'race_type_code', 'cond_code_youngest']
    df_race_info = df_horse_race[race_meta_cols].drop_duplicates()
    df_race = pd.merge(df_race_info, df_race_median, on='race_id')
    
    group_keys = ['course_code', 'track_code', 'distance']
    df_race = pd.merge(df_race, df_base_time[group_keys + ['base_time_seconds']], on=group_keys, how='left')
    
    def compute_pct_deviation(row):
        if pd.isna(row['race_median_time']) or pd.isna(row['base_time_seconds']) or row['base_time_seconds'] <= 0:
            return None
        raw_pct = (row['race_median_time'] - row['base_time_seconds']) / row['base_time_seconds']
        t_cat = row['track_category_for_cond']
        if t_cat in CLASS_RATE_TIME:
            cg = get_class_group(row['cond_code_youngest'])
            class_rate = CLASS_RATE_TIME[t_cat].get(cg, 0.0)
        else:
            class_rate = 0.0
        return raw_pct - class_rate

    df_race['deviation_pct'] = df_race.apply(compute_pct_deviation, axis=1)

    # 2歳新馬戦はバイアス測定から除外
    def is_target_race(row):
        c = str(row['cond_code_youngest']).strip() if pd.notna(row['cond_code_youngest']) else ''
        rt = str(row['race_type_code']).strip() if pd.notna(row['race_type_code']) else ''
        if c == '701' or rt == '11':
            return False
        return True

    df_race_for_bias = df_race[
        df_race.apply(is_target_race, axis=1) & 
        df_race['deviation_pct'].notna()
    ].sort_values(['race_date', 'course_code', 'track_category_for_cond', 'race_number'])
    
    bias_map = {}
    alpha = 1.0  # 縮小推定パラメータ
    for (r_date, c_code, t_cat), grp in df_race_for_bias.groupby(['race_date', 'course_code', 'track_category_for_cond']):
        devs = grp['deviation_pct'].values
        if len(devs) == 0:
            tm = 0.0
        elif len(devs) <= 2:
            tm = float(np.median(devs))
        else:
            tm = float(trim_mean(devs, 0.20))
        N = len(grp)
        bias_pct = (N * tm + alpha * 0.0) / (N + alpha)
        bias_map[(r_date, c_code, t_cat)] = bias_pct

    rows = []
    for _, r in df_race.iterrows():
        key = (r['race_date'], r['course_code'], r['track_category_for_cond'])
        bias_pct = bias_map.get(key, 0.0)
        bt = r['base_time_seconds']
        bias_sec = (bt * bias_pct) if pd.notna(bt) else 0.0
        rows.append({
            'race_id': r['race_id'],
            'race_date': r['race_date'],
            'course_code': r['course_code'],
            'track_code': r['track_code'],
            'track_bias_seconds': bias_sec
        })
            
    df_bias = pd.DataFrame(rows).drop_duplicates(subset=['race_id'])
    return df_bias[['race_id', 'race_date', 'course_code', 'track_code', 'track_bias_seconds']]

def calculate_track_bias_3f(df_horse_race: pd.DataFrame, df_base_time_3f: pd.DataFrame) -> pd.DataFrame:
    """
    各レースの上がり3F中央値と統一基準3Fタイムの乖離から、上がりトラックバイアスを算出。
    - パーセンテージ偏差に換算してトリム平均（20%）を行う。
    - レース固有のクラス差を控除して純粋な上がりバイアス率を抽出。
    - 2歳新馬戦はバイアス測定から除外。
    """
    df_valid = df_horse_race[df_horse_race['last_3f_time_seconds'] > 0].copy()
    df_race_median = df_valid.groupby('race_id')['last_3f_time_seconds'].median().reset_index(name='race_median_3f')
    
    race_meta_cols = ['race_id', 'race_date', 'course_code', 'track_code', 'track_category_for_cond', 'distance', 'race_number', 'race_type_code', 'cond_code_youngest']
    df_race_info = df_horse_race[race_meta_cols].drop_duplicates()
    df_race = pd.merge(df_race_info, df_race_median, on='race_id')
    
    group_keys = ['course_code', 'track_code', 'distance']
    df_race = pd.merge(df_race, df_base_time_3f[group_keys + ['base_3f_time_seconds']], on=group_keys, how='left')
    
    def compute_3f_pct_deviation(row):
        if pd.isna(row['race_median_3f']) or pd.isna(row['base_3f_time_seconds']) or row['base_3f_time_seconds'] <= 0:
            return None
        raw_pct = (row['race_median_3f'] - row['base_3f_time_seconds']) / row['base_3f_time_seconds']
        t_cat = row['track_category_for_cond']
        if t_cat in CLASS_RATE_3F:
            cg = get_class_group(row['cond_code_youngest'])
            class_rate = CLASS_RATE_3F[t_cat].get(cg, 0.0)
        else:
            class_rate = 0.0
        return raw_pct - class_rate

    df_race['deviation_3f_pct'] = df_race.apply(compute_3f_pct_deviation, axis=1)

    def is_target_race(row):
        c = str(row['cond_code_youngest']).strip() if pd.notna(row['cond_code_youngest']) else ''
        rt = str(row['race_type_code']).strip() if pd.notna(row['race_type_code']) else ''
        if c == '701' or rt == '11':
            return False
        return True

    df_race_for_bias = df_race[
        df_race.apply(is_target_race, axis=1) & 
        df_race['deviation_3f_pct'].notna()
    ].sort_values(['race_date', 'course_code', 'track_category_for_cond', 'race_number'])
    
    bias_3f_map = {}
    alpha = 1.0
    for (r_date, c_code, t_cat), grp in df_race_for_bias.groupby(['race_date', 'course_code', 'track_category_for_cond']):
        devs = grp['deviation_3f_pct'].values
        if len(devs) == 0:
            tm = 0.0
        elif len(devs) <= 2:
            tm = float(np.median(devs))
        else:
            tm = float(trim_mean(devs, 0.20))
        N = len(grp)
        bias_pct = (N * tm + alpha * 0.0) / (N + alpha)
        bias_3f_map[(r_date, c_code, t_cat)] = bias_pct

    rows = []
    for _, r in df_race.iterrows():
        key = (r['race_date'], r['course_code'], r['track_category_for_cond'])
        bias_pct = bias_3f_map.get(key, 0.0)
        b3f = r['base_3f_time_seconds']
        bias_3f_sec = (b3f * bias_pct) if pd.notna(b3f) else 0.0
        rows.append({
            'race_id': r['race_id'],
            'race_date': r['race_date'],
            'course_code': r['course_code'],
            'track_code': r['track_code'],
            'track_bias_3f_seconds': bias_3f_sec
        })
            
    df_bias_3f = pd.DataFrame(rows).drop_duplicates(subset=['race_id'])
    return df_bias_3f[['race_id', 'race_date', 'course_code', 'track_code', 'track_bias_3f_seconds']]

def calculate_ucv(df_horse_race: pd.DataFrame, df_base_time: pd.DataFrame, df_track_bias: pd.DataFrame) -> pd.DataFrame:
    """
    各競走馬の走破タイムUCVスコア（Zスコア）を算出。
    統一基準タイムとの乖離を標準偏差で正規化し、外れ値クリッピング（-4.0〜+4.0σ、偏差値10〜90）を適用。
    """
    group_keys = ['course_code', 'track_code', 'distance']
    df_ucv = pd.merge(df_horse_race, df_base_time, on=group_keys, how='left')
    df_ucv = pd.merge(df_ucv, df_track_bias[['race_id', 'track_bias_seconds']], on='race_id', how='left')
    df_ucv['track_bias_seconds'] = df_ucv['track_bias_seconds'].fillna(0.0)
    df_ucv['corrected_time'] = df_ucv['time_seconds'] - df_ucv['track_bias_seconds']
    
    def compute_z_score(row):
        if pd.isna(row['base_time_seconds']) or pd.isna(row['corrected_time']): return None
        if pd.isna(row['std_dev']) or row['std_dev'] <= 0: return 0.0
        z = (row['base_time_seconds'] - row['corrected_time']) / row['std_dev']
        return max(min(z, 4.0), -4.0)

    df_ucv['ucv_score'] = df_ucv.apply(compute_z_score, axis=1)
    return df_ucv[['race_id', 'horse_number', 'time_seconds', 'track_bias_seconds', 'base_time_seconds', 'corrected_time', 'ucv_score']]

def calculate_ucv_3f(df_horse_race: pd.DataFrame, df_base_time_3f: pd.DataFrame, df_track_bias_3f: pd.DataFrame) -> pd.DataFrame:
    """
    各競走馬の上がり3F UCVスコア（Zスコア）を算出。
    統一基準上がり3Fタイムとの乖離を正規化し、外れ値クリッピング（-4.0〜+4.0σ）を適用。
    """
    group_keys = ['course_code', 'track_code', 'distance']
    df_ucv_3f = pd.merge(df_horse_race, df_base_time_3f, on=group_keys, how='left')
    df_ucv_3f = pd.merge(df_ucv_3f, df_track_bias_3f[['race_id', 'track_bias_3f_seconds']], on='race_id', how='left')
    
    df_ucv_3f['track_bias_3f_seconds'] = df_ucv_3f['track_bias_3f_seconds'].fillna(0.0)
    df_ucv_3f['corrected_3f_time'] = df_ucv_3f['last_3f_time_seconds'] - df_ucv_3f['track_bias_3f_seconds']
    
    def compute_3f_z_score(row):
        if pd.isna(row['base_3f_time_seconds']) or pd.isna(row['corrected_3f_time']) or row['last_3f_time_seconds'] <= 0:
            return None
        if pd.isna(row['base_3f_std']) or row['base_3f_std'] <= 0:
            return 0.0
        z = (row['base_3f_time_seconds'] - row['corrected_3f_time']) / row['base_3f_std']
        return max(min(z, 4.0), -4.0)

    df_ucv_3f['ucv_3f_score'] = df_ucv_3f.apply(compute_3f_z_score, axis=1)
    return df_ucv_3f[['race_id', 'horse_number', 'last_3f_time_seconds', 'track_bias_3f_seconds', 'base_3f_time_seconds', 'corrected_3f_time', 'ucv_3f_score']]

def save_to_database(conn: sqlite3.Connection, df_base_time: pd.DataFrame, df_track_bias: pd.DataFrame, df_ucv: pd.DataFrame, 
                     df_base_time_3f: pd.DataFrame, df_track_bias_3f: pd.DataFrame, df_ucv_3f: pd.DataFrame):
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

    print("   -> 統一基準タイムマスタを保存しています...")
    df_base_time.to_sql('ucv_base_time', conn, if_exists='append', index=False, method='multi', chunksize=500)
    df_track_bias.to_sql('ucv_track_bias', conn, if_exists='append', index=False, method='multi', chunksize=500)
    df_base_time_3f.to_sql('ucv_3f_base_time', conn, if_exists='append', index=False, method='multi', chunksize=500)
    df_track_bias_3f.to_sql('ucv_3f_track_bias', conn, if_exists='append', index=False, method='multi', chunksize=500)
    
    print("   -> 全頭UCVスコアを保存しています...")
    df_ucv = df_ucv.drop_duplicates(subset=['race_id', 'horse_number'], keep='last').replace({np.nan: None})
    data_tuples_ucv = list(df_ucv.itertuples(index=False, name=None))
    sql_ucv = """
    INSERT OR REPLACE INTO horse_race_ucv 
    (race_id, horse_number, time_seconds, applied_track_bias, applied_base_time, corrected_time, ucv_score)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    cursor.executemany(sql_ucv, data_tuples_ucv)

    print("   -> 全頭上がり3F UCVスコアを保存しています...")
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
            lap_time_back_3f,
            abnormality_code
        FROM horse_race_info
        WHERE abnormality_code IN ('0', '7')
          AND horse_number != '00'
          AND running_time != '0000'
    """
    
    df_race = pd.read_sql(query_race, conn)
    df_horse = pd.read_sql(query_horse, conn)
    
    print("2. データの整形・時間パースを行っています...")
    df_horse['time_seconds'] = df_horse['running_time'].apply(parse_running_time)
    df_horse['last_3f_time_seconds'] = df_horse['lap_time_back_3f'].apply(parse_3f_time)
    
    df_horse_race = pd.merge(df_horse, df_race, on='race_id', how='inner')
    df_horse_race = df_horse_race.dropna(subset=['time_seconds'])
    df_horse_race = df_horse_race[df_horse_race['time_seconds'] >= 50.0]
    
    df_horse_race['track_category_for_cond'] = df_horse_race['track_code'].apply(get_track_category)
    
    def is_good(row):
        cat = row['track_category_for_cond']
        if cat == '芝': return str(row['turf_condition_code']).strip() == '1'
        if cat == 'ダート': return str(row['dirt_condition_code']).strip() == '1'
        return False
        
    df_horse_race['is_good_condition'] = df_horse_race.apply(is_good, axis=1)
    
    print("3. 全コースの統一基準タイムマスタ(走破タイム & 上がり3F)を算出中...")
    df_base_time = generate_base_time_master(df_horse_race)
    df_base_time_3f = calculate_ucv_3f_base_time(df_horse_race)
    
    print(f"   -> 統一基準タイム作成件数: 走破タイム {len(df_base_time)}コース, 上がり3F {len(df_base_time_3f)}コース")
    
    print("4. トリム平均(0.20)による馬場差(Track Bias)を全自動算出中...")
    df_track_bias = calculate_track_bias(df_horse_race, df_base_time)
    df_track_bias_3f = calculate_track_bias_3f(df_horse_race, df_base_time_3f)
    
    print("5. 全競走馬の統一UCVスコアおよび上がり3F UCVスコアを計算中...")
    df_ucv = calculate_ucv(df_horse_race, df_base_time, df_track_bias)
    df_ucv_3f = calculate_ucv_3f(df_horse_race, df_base_time_3f, df_track_bias_3f)
    
    print("6. データベースへ保存中...")
    save_to_database(conn, df_base_time, df_track_bias, df_ucv, df_base_time_3f, df_track_bias_3f, df_ucv_3f)
    conn.close()
    print("全処理が正常に完了しました！")

if __name__ == '__main__':
    main()

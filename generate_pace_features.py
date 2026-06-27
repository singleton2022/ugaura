import sqlite3
import pandas as pd
import numpy as np
import time

DB_PATH = "C:/Ugaura/sqlite/jra_race.db"

def calculate_and_save_features():
    total_start = time.time()
    
    print("1. 生データの抽出を開始...")
    t0 = time.time()
    query = """
    SELECT 
        rd.year || rd.month_day || rd.course_code || rd.times || rd.day || rd.race_number AS race_id,
        rd.year || '-' || substr(rd.month_day, 1, 2) || '-' || substr(rd.month_day, 3, 2) AS race_date,
        hri.blood_reg_number AS horse_id,
        CAST(rd.started_count AS INTEGER) AS started_count,
        hri.corner_1_order,
        hri.corner_2_order,
        hri.corner_3_order,
        hri.corner_4_order,
        hri.abnormality_code
    FROM race_detail rd
    JOIN horse_race_info hri 
        ON rd.year = hri.year AND rd.month_day = hri.month_day 
        AND rd.course_code = hri.course_code AND rd.times = hri.times 
        AND rd.day = hri.day AND rd.race_number = hri.race_number
    """
    
    # コネクションを明示的に開閉
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(query, conn)
    conn.close()

    df = df.sort_values(by='race_date', ascending=True)
    df = df.drop_duplicates(subset=['race_id', 'horse_id'], keep='last')
    print(f"  -> 完了: {len(df)}件取得 (所要時間: {time.time() - t0:.2f}秒)")

    print("2. 単一レースにおける「初角ポジション指数」の算出...")
    t0 = time.time()
    c1 = pd.to_numeric(df['corner_1_order'], errors='coerce')
    c2 = pd.to_numeric(df['corner_2_order'], errors='coerce')
    c3 = pd.to_numeric(df['corner_3_order'], errors='coerce')
    c4 = pd.to_numeric(df['corner_4_order'], errors='coerce')

    c1 = c1.where(c1 > 0, np.nan)
    c2 = c2.where(c2 > 0, np.nan)
    c3 = c3.where(c3 > 0, np.nan)
    c4 = c4.where(c4 > 0, np.nan)

    first_corner = c1.combine_first(c2).combine_first(c3).combine_first(c4)
    valid_mask = df['abnormality_code'].isin(['0', '7'])
    df['first_corner_pos'] = np.where(valid_mask, first_corner, np.nan)
    
    df['raw_dash_score'] = np.where(
        (df['started_count'] > 1) & (df['first_corner_pos'].notna()),
        (df['first_corner_pos'] - 1) / (df['started_count'] - 1),
        np.nan
    )
    print(f"  -> 完了 (所要時間: {time.time() - t0:.2f}秒)")

    print("3. 各馬の「過去5戦のダッシュ力中央値」を算出...")
    t0 = time.time()
    df = df.sort_values(by=['horse_id', 'race_date'])
    df['dash_score_median'] = df.groupby('horse_id')['raw_dash_score'] \
                                .rolling(window=5, min_periods=1, closed='left') \
                                .median() \
                                .reset_index(level=0, drop=True)
    
    df['dash_score_median'] = df['dash_score_median'].fillna(0.5)
    df_dash_out = df[['race_id', 'horse_id', 'dash_score_median']].copy()
    print(f"  -> 完了 (所要時間: {time.time() - t0:.2f}秒)")

    print("4. レース全体の「ペース予測スコア」の算出...")
    t0 = time.time()
    pace_features = []
    
    grouped = df.groupby('race_id')
    total_races = len(grouped)
    print(f"  -> 対象レース数: {total_races}件")
    
    for i, (race_id, group) in enumerate(grouped):
        if i > 0 and i % 20000 == 0:
            print(f"  -> 処理中: {i} / {total_races} レース完了...")

        scores = group['dash_score_median'].sort_values().values
        if len(scores) < 3:
            continue
            
        pace_features.append({
            'race_id': race_id,
            'pace_score_top3': float(scores[:3].mean()),
            'leader_margin': float(scores[1] - scores[0])
        })
        
    df_pace_out = pd.DataFrame(pace_features)
    print(f"  -> 完了 (所要時間: {time.time() - t0:.2f}秒)")

    print("5. SQLiteデータベースへの保存...")
    t0 = time.time()
    
    conn = sqlite3.connect(DB_PATH)
    try:
        # WALモード設定を削除し、同期モードのみ調整
        conn.execute("PRAGMA synchronous = NORMAL;")
        
        conn.execute("DELETE FROM feature_dash_score")
        dash_records = df_dash_out.to_records(index=False).tolist()
        conn.executemany(
            "INSERT OR REPLACE INTO feature_dash_score (race_id, horse_id, dash_score_median) VALUES (?, ?, ?)",
            dash_records
        )
        print("  -> feature_dash_score 保存完了")
        
        conn.execute("DELETE FROM feature_pace_prediction")
        pace_records = df_pace_out.to_records(index=False).tolist()
        conn.executemany(
            "INSERT OR REPLACE INTO feature_pace_prediction (race_id, pace_score_top3, leader_margin) VALUES (?, ?, ?)",
            pace_records
        )
        print("  -> feature_pace_prediction 保存完了")

        conn.commit()
    finally:
        conn.close()

    print(f"完了: ダッシュスコア {len(df_dash_out)}件、ペーススコア {len(df_pace_out)}件を保存しました。(保存フェーズ所要時間: {time.time() - t0:.2f}秒)")
    print(f"総所要時間: {time.time() - total_start:.2f}秒")

if __name__ == "__main__":
    calculate_and_save_features()
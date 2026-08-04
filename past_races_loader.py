import sqlite3
import pandas as pd
import numpy as np

COURSE_MAP = {
    '01': '札幌', '02': '函館', '03': '福島', '04': '新潟', '05': '東京',
    '06': '中山', '07': '中京', '08': '京都', '09': '阪神', '10': '小倉'
}

def load_past_races_for_horses(horse_id_list, current_race_date, db_path='C:/Ugaura/sqlite/jra_race.db'):
    if not horse_id_list:
        return pd.DataFrame()

    conn = sqlite3.connect(db_path)
    placeholders = ','.join(['?'] * len(horse_id_list))
    
    query = f"""
    WITH horse_history AS (
        SELECT 
            hri.blood_reg_number AS horse_id,
            hri.horse_name,
            hri.horse_number,
            hri.jockey_name_short,
            (rd.year || rd.month_day || rd.course_code || rd.times || rd.day || rd.race_number) AS past_race_id,
            rd.year || '-' || substr(rd.month_day, 1, 2) || '-' || substr(rd.month_day, 3, 2) AS race_date,
            rd.course_code,
            rd.track_code,
            rd.distance,
            rd.grade_code,
            rd.cond_code_youngest,
            hri.final_order,
            hri.running_time AS race_time,
            hri.lap_time_back_3f AS last_3f,
            ucv.ucv_score,
            u3f.ucv_3f_score,
            ROW_NUMBER() OVER (
                PARTITION BY hri.blood_reg_number 
                ORDER BY rd.year DESC, rd.month_day DESC
            ) AS rn
        FROM horse_race_info hri
        JOIN race_detail rd 
            ON rd.year = hri.year AND rd.month_day = hri.month_day AND rd.course_code = hri.course_code 
            AND rd.times = hri.times AND rd.day = hri.day AND rd.race_number = hri.race_number
        LEFT JOIN horse_race_ucv ucv 
            ON (rd.year || rd.month_day || rd.course_code || rd.times || rd.day || rd.race_number) = ucv.race_id 
            AND hri.horse_number = ucv.horse_number
        LEFT JOIN horse_race_ucv_3f u3f 
            ON (rd.year || rd.month_day || rd.course_code || rd.times || rd.day || rd.race_number) = u3f.race_id 
            AND hri.horse_number = u3f.horse_number
        WHERE hri.blood_reg_number IN ({placeholders})
          AND (rd.year || '-' || substr(rd.month_day, 1, 2) || '-' || substr(rd.month_day, 3, 2)) < ?
          AND hri.abnormality_code IN ('0', '7')
          AND hri.horse_number != '00'
          AND hri.running_time != '0000'
    )
    SELECT * FROM horse_history WHERE rn <= 5 ORDER BY horse_id, rn;
    """
    
    params = list(horse_id_list) + [current_race_date]
    df = pd.read_sql(query, conn, params=params)

    if df.empty:
        conn.close()
        return df

    df['course_name'] = df['course_code'].astype(str).str.zfill(2).map(COURSE_MAP).fillna('その他')
    df['track_type'] = np.where(pd.to_numeric(df['track_code'], errors='coerce').between(10, 22), '芝', 'ダート')
    df['past_label'] = df['rn'].astype(str) + '走前'
    
    def format_time(t):
        if pd.isna(t) or t == 0:
            return '-'
        t_sec = float(t) / 10.0
        m = int(t_sec // 60)
        s = t_sec % 60
        return f"{m}:{s:04.1f}" if m > 0 else f"{s:.1f}"

    def format_3f(t):
        if pd.isna(t) or t == 0:
            return '-'
        return f"{float(t)/10.0:.1f}"

    df['time_fmt'] = pd.to_numeric(df['race_time'], errors='coerce').apply(format_time)
    df['last_3f_fmt'] = pd.to_numeric(df['last_3f'], errors='coerce').apply(format_3f)

    # 【自分を除外した他馬の次走3着以内頭数を一括計算】
    # 次走の対象期間は過去走日より後、かつ当日のレース日より前 (strict less than)
    unique_past_ids = df['past_race_id'].unique().tolist()
    past_placeholders = ','.join(['?'] * len(unique_past_ids))

    query_next_races = f"""
    WITH target_past_races AS (
        SELECT DISTINCT 
            (rd.year || rd.month_day || rd.course_code || rd.times || rd.day || rd.race_number) AS past_race_id,
            rd.year || '-' || substr(rd.month_day, 1, 2) || '-' || substr(rd.month_day, 3, 2) AS past_race_date
        FROM race_detail rd
        WHERE (rd.year || rd.month_day || rd.course_code || rd.times || rd.day || rd.race_number) IN ({past_placeholders})
    ),
    past_race_horses AS (
        SELECT 
            tpr.past_race_id,
            tpr.past_race_date,
            hri.blood_reg_number AS horse_id
        FROM target_past_races tpr
        JOIN horse_race_info hri 
            ON tpr.past_race_id = (hri.year || hri.month_day || hri.course_code || hri.times || hri.day || hri.race_number)
        WHERE hri.abnormality_code IN ('0', '7')
          AND hri.horse_number != '00'
    ),
    next_races AS (
        SELECT 
            prh.past_race_id,
            prh.horse_id,
            hri.final_order,
            ROW_NUMBER() OVER (
                PARTITION BY prh.past_race_id, prh.horse_id
                ORDER BY rd.year ASC, rd.month_day ASC, rd.race_number ASC
            ) AS rn
        FROM past_race_horses prh
        JOIN horse_race_info hri ON prh.horse_id = hri.blood_reg_number
        JOIN race_detail rd 
            ON rd.year = hri.year AND rd.month_day = hri.month_day AND rd.course_code = hri.course_code 
            AND rd.times = hri.times AND rd.day = hri.day AND rd.race_number = hri.race_number
        WHERE (rd.year || '-' || substr(rd.month_day, 1, 2) || '-' || substr(rd.month_day, 3, 2)) > prh.past_race_date
          AND (rd.year || '-' || substr(rd.month_day, 1, 2) || '-' || substr(rd.month_day, 3, 2)) < ?
          AND hri.abnormality_code IN ('0', '7')
          AND hri.running_time != '0000'
    )
    SELECT 
        past_race_id,
        horse_id,
        CASE WHEN CAST(final_order AS INTEGER) BETWEEN 1 AND 3 THEN 1 ELSE 0 END AS self_next_top3
    FROM next_races
    WHERE rn = 1;
    """
    
    next_df = pd.read_sql(query_next_races, conn, params=list(unique_past_ids) + [current_race_date])
    conn.close()

    if not next_df.empty:
        total_top3_by_race = next_df.groupby('past_race_id')['self_next_top3'].agg(
            total_top3='sum',
            total_next_ran='count'
        ).to_dict(orient='index')

        self_flag_map = next_df.set_index(['past_race_id', 'horse_id'])['self_next_top3'].to_dict()

        other_top3_counts = []
        other_next_rans = []

        for _, row in df.iterrows():
            p_id = str(row['past_race_id'])
            h_id = str(row['horse_id'])
            
            race_stats = total_top3_by_race.get(p_id, {'total_top3': 0, 'total_next_ran': 0})
            total_t3 = race_stats['total_top3']
            total_ran = race_stats['total_next_ran']
            
            self_ran = 1 if (p_id, h_id) in self_flag_map else 0
            self_t3 = self_flag_map.get((p_id, h_id), 0)
            
            other_t3 = max(0, total_t3 - self_t3)
            other_ran = max(0, total_ran - self_ran)
            
            other_top3_counts.append(other_t3)
            other_next_rans.append(other_ran)

        df['other_next_top3_count'] = other_top3_counts
        df['other_next_ran'] = other_next_rans
    else:
        df['other_next_top3_count'] = 0
        df['other_next_ran'] = 0

    return df

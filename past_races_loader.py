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
            rd.year || '-' || substr(rd.month_day, 1, 2) || '-' || substr(rd.month_day, 3, 2) AS race_date,
            rd.course_code,
            rd.track_code,
            rd.distance,
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
    )
    SELECT * FROM horse_history WHERE rn <= 5 ORDER BY horse_id, rn;
    """
    
    params = list(horse_id_list) + [current_race_date]
    df = pd.read_sql(query, conn, params=params)
    conn.close()

    if df.empty:
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
    df['ucv_score'] = pd.to_numeric(df['ucv_score'], errors='coerce').round(2)
    df['ucv_3f_score'] = pd.to_numeric(df['ucv_3f_score'], errors='coerce').round(2)

    return df

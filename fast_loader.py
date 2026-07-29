import sqlite3
import pandas as pd
import numpy as np

COURSE_MAP = {
    '01': '札幌', '02': '函館', '03': '福島', '04': '新潟', '05': '東京',
    '06': '中山', '07': '中京', '08': '京都', '09': '阪神', '10': '小倉'
}

def get_available_dates(db_path='C:/Ugaura/sqlite/jra_race.db'):
    conn = sqlite3.connect(db_path)
    query = """
    SELECT DISTINCT 
        rd.year || '-' || substr(rd.month_day, 1, 2) || '-' || substr(rd.month_day, 3, 2) AS race_date 
    FROM race_detail rd 
    WHERE rd.year >= '2024' 
    ORDER BY race_date DESC;
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df['race_date'].tolist()

def get_races_for_date(selected_date, db_path='C:/Ugaura/sqlite/jra_race.db'):
    conn = sqlite3.connect(db_path)
    query = """
    SELECT DISTINCT 
        (rd.year || rd.month_day || rd.course_code || rd.times || rd.day || rd.race_number) AS race_id,
        rd.course_code,
        CAST(rd.race_number AS INTEGER) AS race_number,
        rd.distance,
        rd.track_code
    FROM race_detail rd
    WHERE (rd.year || '-' || substr(rd.month_day, 1, 2) || '-' || substr(rd.month_day, 3, 2)) = ?
    ORDER BY rd.course_code, race_number;
    """
    df = pd.read_sql(query, conn, params=[selected_date])
    conn.close()
    
    if df.empty:
        return {}

    df['course_name'] = df['course_code'].astype(str).str.zfill(2).map(COURSE_MAP).fillna('その他')
    df['track_type'] = np.where(pd.to_numeric(df['track_code'], errors='coerce').between(10, 22), '芝', 'ダート')
    df['race_display'] = df['course_name'] + ' ' + df['race_number'].astype(str) + 'R (' + df['track_type'] + df['distance'].astype(str) + 'm)'
    
    return df.set_index('race_id')['race_display'].to_dict()

def load_single_race_card(race_id, db_path='C:/Ugaura/sqlite/jra_race.db'):
    conn = sqlite3.connect(db_path)
    query = """
    WITH target_horses AS (
        SELECT 
            (rd.year || rd.month_day || rd.course_code || rd.times || rd.day || rd.race_number) AS race_id,
            rd.year || '-' || substr(rd.month_day, 1, 2) || '-' || substr(rd.month_day, 3, 2) AS race_date,
            rd.course_code,
            CAST(rd.race_number AS INTEGER) AS race_number,
            rd.distance,
            rd.track_code,
            rd.turf_condition_code,
            rd.dirt_condition_code,
            hri.horse_number,
            hri.horse_name,
            hri.blood_reg_number AS horse_id,
            hri.bracket_number,
            hri.sex_code,
            hri.horse_age,
            CAST(hri.weight_carried AS REAL) / 10.0 AS weight_carried,
            hri.jockey_name_short,
            hri.jockey_code,
            hri.horse_weight,
            hri.weight_change_sign,
            hri.weight_change,
            hri.win_odds,
            hri.corner_4_order,
            hri.running_style,
            CASE WHEN CAST(hri.final_order AS INTEGER) = 1 THEN 1 ELSE 0 END AS is_win
        FROM race_detail rd
        JOIN horse_race_info hri 
            ON rd.year = hri.year AND rd.month_day = hri.month_day AND rd.course_code = hri.course_code 
            AND rd.times = hri.times AND rd.day = hri.day AND rd.race_number = hri.race_number
        WHERE (rd.year || rd.month_day || rd.course_code || rd.times || rd.day || rd.race_number) = ?
    ),
    current_jockeys AS (
        SELECT DISTINCT jockey_code FROM target_horses WHERE jockey_code IS NOT NULL AND jockey_code != ''
    ),
    jockey_stats AS (
        SELECT 
            hri.jockey_code,
            ROUND(AVG(CASE WHEN CAST(hri.final_order AS INTEGER) BETWEEN 1 AND 3 THEN 1.0 ELSE 0.0 END), 3) AS jockey_top3_rate_100,
            ROUND(AVG(CASE WHEN CAST(hri.final_order AS INTEGER) = 1 THEN 1.0 ELSE 0.0 END), 3) AS jockey_win_rate_100
        FROM (
            SELECT 
                hri.jockey_code, 
                hri.final_order, 
                ROW_NUMBER() OVER (
                    PARTITION BY hri.jockey_code 
                    ORDER BY rd.year DESC, rd.month_day DESC
                ) AS rn
            FROM horse_race_info hri 
            JOIN race_detail rd 
                ON rd.year = hri.year AND rd.month_day = hri.month_day AND rd.course_code = hri.course_code 
                AND rd.times = hri.times AND rd.day = hri.day AND rd.race_number = hri.race_number
            WHERE hri.jockey_code IN (SELECT jockey_code FROM current_jockeys)
              AND (rd.year || '-' || substr(rd.month_day, 1, 2) || '-' || substr(rd.month_day, 3, 2)) < (SELECT MAX(race_date) FROM target_horses)
              AND hri.abnormality_code IN ('0', '7')
        ) hri
        WHERE rn <= 100
        GROUP BY hri.jockey_code
    ),
    history AS (
        SELECT 
            (rd.year || rd.month_day || rd.course_code || rd.times || rd.day || rd.race_number) AS race_id,
            rd.year || '-' || substr(rd.month_day, 1, 2) || '-' || substr(rd.month_day, 3, 2) AS race_date,
            hri.blood_reg_number AS horse_id,
            ucv.ucv_score,
            u3f.ucv_3f_score
        FROM race_detail rd
        JOIN horse_race_info hri 
            ON rd.year = hri.year AND rd.month_day = hri.month_day AND rd.course_code = hri.course_code 
            AND rd.times = hri.times AND rd.day = hri.day AND rd.race_number = hri.race_number
        LEFT JOIN horse_race_ucv ucv 
            ON (rd.year || rd.month_day || rd.course_code || rd.times || rd.day || rd.race_number) = ucv.race_id 
            AND hri.horse_number = ucv.horse_number
        LEFT JOIN horse_race_ucv_3f u3f 
            ON (rd.year || rd.month_day || rd.course_code || rd.times || rd.day || rd.race_number) = u3f.race_id 
            AND hri.horse_number = u3f.horse_number
        WHERE hri.blood_reg_number IN (SELECT horse_id FROM target_horses)
          AND hri.abnormality_code IN ('0', '7')
    ),
    history_rolling AS (
        SELECT 
            race_id,
            horse_id,
            LAG(ucv_score, 1) OVER (PARTITION BY horse_id ORDER BY race_date ASC, race_id ASC) AS prev_ucv_score,
            MAX(ucv_score) OVER (PARTITION BY horse_id ORDER BY race_date ASC, race_id ASC ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS max_ucv_score_5,
            AVG(ucv_score) OVER (PARTITION BY horse_id ORDER BY race_date ASC, race_id ASC ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS avg_ucv_score_5,
            MAX(ucv_3f_score) OVER (PARTITION BY horse_id ORDER BY race_date ASC, race_id ASC ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS max_ucv_3f_5,
            AVG(ucv_3f_score) OVER (PARTITION BY horse_id ORDER BY race_date ASC, race_id ASC ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS avg_ucv_3f_5
        FROM history
    )
    SELECT 
        th.*,
        hr.prev_ucv_score,
        hr.max_ucv_score_5,
        hr.avg_ucv_score_5,
        hr.max_ucv_3f_5,
        hr.avg_ucv_3f_5,
        ds.dash_score_median,
        rt.elo_rating_turf,
        rt.elo_rating_dirt,
        COALESCE(js.jockey_top3_rate_100, 0.25) AS jockey_top3_rate_100,
        COALESCE(js.jockey_win_rate_100, 0.10) AS jockey_win_rate_100
    FROM target_horses th
    LEFT JOIN history_rolling hr ON th.race_id = hr.race_id AND th.horse_id = hr.horse_id
    LEFT JOIN feature_dash_score ds ON th.race_id = ds.race_id AND th.horse_id = ds.horse_id
    LEFT JOIN feature_opponent_rating rt ON th.race_id = rt.race_id AND th.horse_id = rt.horse_id
    LEFT JOIN jockey_stats js ON th.jockey_code = js.jockey_code
    ORDER BY CAST(th.horse_number AS INTEGER);
    """
    df = pd.read_sql(query, conn, params=[race_id])
    conn.close()
    
    if df.empty:
        return df

    df['course_name'] = df['course_code'].astype(str).str.zfill(2).map(COURSE_MAP).fillna('その他')
    df['track_type'] = np.where(pd.to_numeric(df['track_code'], errors='coerce').between(10, 22), '芝', 'ダート')
    df['target_elo_rating'] = np.where(df['track_type'] == '芝', df['elo_rating_turf'], df['elo_rating_dirt'])
    df['avg_ucv_score_5'] = df['avg_ucv_score_5'].fillna(0.0)
    df['avg_ucv_3f_5'] = df['avg_ucv_3f_5'].fillna(0.0)

    return df

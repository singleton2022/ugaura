import sqlite3
import pandas as pd

def get_available_dates(db_path='C:/Ugaura/sqlite/jra_race.db'):
    conn = sqlite3.connect(db_path)
    query = """
    SELECT DISTINCT 
        rd.year || '-' || substr(rd.month_day, 1, 2) || '-' || substr(rd.month_day, 3, 2) AS race_date 
    FROM race_detail rd 
    WHERE rd.year >= '2024'
      AND rd.course_code IN ('01', '02', '03', '04', '05', '06', '07', '08', '09', '10')
    ORDER BY race_date DESC;
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df['race_date'].tolist()

def get_races_for_date(selected_date, db_path='C:/Ugaura/sqlite/jra_race.db'):
    conn = sqlite3.connect(db_path)
    clean_date = selected_date.replace('-', '')
    year_str = clean_date[:4]
    month_day_str = clean_date[4:]

    query = """
    SELECT 
        rd.year || rd.month_day || rd.course_code || rd.times || rd.day || rd.race_number AS race_id,
        rd.course_code,
        rd.race_number,
        rd.race_name_main,
        rd.race_name_short_10,
        rd.cond_code_youngest,
        rd.grade_code,
        rd.track_code,
        rd.distance
    FROM race_detail rd
    WHERE rd.year = ? AND rd.month_day = ?
      AND rd.course_code IN ('01', '02', '03', '04', '05', '06', '07', '08', '09', '10')
    ORDER BY rd.course_code, CAST(rd.race_number AS INTEGER);
    """
    df = pd.read_sql(query, conn, params=[year_str, month_day_str])
    conn.close()

    course_map = {
        '01': '札幌', '02': '函館', '03': '福島', '04': '新潟', '05': '東京',
        '06': '中山', '07': '中京', '08': '京都', '09': '阪神', '10': '小倉'
    }
    
    def get_class_label(cond, grade):
        cond = str(cond).strip() if cond is not None else ''
        grade = str(grade).strip() if grade is not None else ''
        if grade in ['A', 'B', 'C', 'D', 'E', 'G', 'H', 'L']:
            return 'OP/重賞'
        elif cond in ['701', '702', '703']:
            return '未勝利'
        elif cond in ['003', '004', '005']:
            return '1勝クラス'
        elif cond in ['007', '008', '009', '010']:
            return '2勝クラス'
        elif cond in ['014', '015', '016']:
            return '3勝クラス'
        elif cond in ['000', '999']:
            return 'オープン'
        return '一般'

    races = {}
    for _, row in df.iterrows():
        c_name = course_map.get(str(row['course_code']).zfill(2), '他')
        r_num = int(row['race_number'])
        
        raw_short_name = str(row['race_name_short_10']).strip() if pd.notna(row['race_name_short_10']) else ''
        raw_main_name = str(row['race_name_main']).strip() if pd.notna(row['race_name_main']) else ''
        
        if raw_short_name:
            race_name_disp = raw_short_name
        elif raw_main_name:
            race_name_disp = raw_main_name
        else:
            race_name_disp = get_class_label(row['cond_code_youngest'], row['grade_code'])

        t_code = pd.to_numeric(row['track_code'], errors='coerce')
        t_type = '芝' if 10 <= t_code <= 22 else 'ダ'
        dist = f"{t_type}{row['distance']}m"
        
        display_str = f"{c_name} {r_num}R {race_name_disp} ({dist})"
        races[row['race_id']] = display_str

    return races

def load_single_race_card(race_id, db_path='C:/Ugaura/sqlite/jra_race.db'):
    conn = sqlite3.connect(db_path)
    query = """
    SELECT 
        (rd.year || rd.month_day || rd.course_code || rd.times || rd.day || rd.race_number) AS race_id,
        rd.year || '-' || substr(rd.month_day, 1, 2) || '-' || substr(rd.month_day, 3, 2) AS race_date,
        rd.course_code,
        rd.race_number,
        rd.race_name_main,
        rd.cond_code_youngest,
        rd.grade_code,
        hri.bracket_number,
        hri.horse_number,
        hri.blood_reg_number AS horse_id,
        hri.horse_name,
        hri.sex_code,
        hri.horse_age,
        hri.weight_carried,
        hri.jockey_name_short,
        hri.horse_weight,
        hri.weight_change_sign,
        hri.weight_change,
        hri.win_odds,
        hri.win_popularity
    FROM race_detail rd
    JOIN horse_race_info hri 
        ON rd.year = hri.year AND rd.month_day = hri.month_day AND rd.course_code = hri.course_code 
        AND rd.times = hri.times AND rd.day = hri.day AND rd.race_number = hri.race_number
    WHERE (rd.year || rd.month_day || rd.course_code || rd.times || rd.day || rd.race_number) = ?
      AND hri.abnormality_code IN ('0', '7')
      AND hri.horse_number != '00'
    ORDER BY CAST(hri.horse_number AS INTEGER);
    """
    df = pd.read_sql(query, conn, params=[race_id])
    conn.close()
    return df

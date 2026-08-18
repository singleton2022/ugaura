import sqlite3
import pandas as pd
import streamlit as st

@st.cache_data(show_spinner=False)
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

@st.cache_data(show_spinner=False)
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
        rd.race_name_short_6,
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
            return '重賞'
        elif cond in ['701']:
            return '新馬'
        elif cond in ['702', '703']:
            return '未勝利'
        elif cond in ['003', '004', '005']:
            return '1勝'
        elif cond in ['007', '008', '009', '010']:
            return '2勝'
        elif cond in ['014', '015', '016']:
            return '3勝'
        elif cond in ['000', '999']:
            return 'OP'
        return '一般'

    races = {}
    for _, row in df.iterrows():
        c_name = course_map.get(str(row['course_code']).zfill(2), '他')
        r_num = int(row['race_number'])
        
        raw_short6 = str(row['race_name_short_6']).strip() if pd.notna(row['race_name_short_6']) else ''
        raw_short10 = str(row['race_name_short_10']).strip() if pd.notna(row['race_name_short_10']) else ''
        raw_main_name = str(row['race_name_main']).strip() if pd.notna(row['race_name_main']) else ''
        
        if raw_short6:
            race_name_disp = raw_short6
        elif raw_short10:
            race_name_disp = raw_short10
        elif raw_main_name:
            race_name_disp = raw_main_name
        else:
            race_name_disp = get_class_label(row['cond_code_youngest'], row['grade_code'])

        t_code = pd.to_numeric(row['track_code'], errors='coerce')
        t_type = '芝' if 10 <= t_code <= 22 else 'ダ'
        dist = f"{t_type}{row['distance']}"
        
        display_str = f"{c_name}{r_num}R {race_name_disp} {dist}"
        races[row['race_id']] = display_str

    return races

@st.cache_data(show_spinner=False)
def load_single_race_card(race_id, db_path='C:/Ugaura/sqlite/jra_race.db'):
    conn = sqlite3.connect(db_path)
    clean_id = str(race_id).strip()
    year_str = clean_id[0:4]
    month_day_str = clean_id[4:8]
    course_code_str = clean_id[8:10]
    times_str = clean_id[10:12]
    day_str = clean_id[12:14]
    race_number_str = clean_id[14:16]

    query = """
    SELECT 
        hri.bracket_number,
        hri.horse_number,
        hri.blood_reg_number AS horse_id,
        hri.horse_name,
        hri.sex_code,
        hri.horse_age,
        hri.weight_carried,
        hri.jockey_name_short,
        hri.trainer_name_short,
        hri.win_odds,
        hri.win_popularity,
        hri.horse_weight,
        hri.weight_change_sign,
        hri.weight_change,
        rd.race_number,
        rd.race_name_main,
        rd.race_name_short_10,
        rd.race_name_short_6,
        rd.race_sign_code,
        rd.cond_code_youngest,
        rd.grade_code,
        rd.course_code,
        rd.track_code,
        rd.distance
    FROM horse_race_info hri
    JOIN race_detail rd 
        ON rd.year = hri.year AND rd.month_day = hri.month_day AND rd.course_code = hri.course_code 
        AND rd.times = hri.times AND rd.day = hri.day AND rd.race_number = hri.race_number
    WHERE rd.year = ? AND rd.month_day = ? AND rd.course_code = ? 
      AND rd.times = ? AND rd.day = ? AND rd.race_number = ?
      AND hri.abnormality_code IN ('0', '7')
      AND hri.horse_number != '00'
    ORDER BY CAST(hri.horse_number AS INTEGER);
    """

    params = [year_str, month_day_str, course_code_str, times_str, day_str, race_number_str]
    df = pd.read_sql(query, conn, params=params)
    conn.close()

    return df

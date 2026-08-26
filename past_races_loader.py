import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
from scipy.stats import trim_mean

COURSE_MAP = {
    '01': '札幌', '02': '函館', '03': '福島', '04': '新潟', '05': '東京',
    '06': '中山', '07': '中京', '08': '京都', '09': '阪神', '10': '小倉'
}

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

def get_class_info(cond_code, grade_code, course_code=None):
    cond = str(cond_code).strip() if cond_code is not None else ''
    grade = str(grade_code).strip() if grade_code is not None else ''
    c_code = str(course_code).strip() if course_code is not None else ''
    
    is_local = False
    if c_code:
        try:
            c_num = int(c_code)
            if c_num not in range(1, 11):
                is_local = True
        except ValueError:
            pass

    if grade == 'A':
        return 5.0, 'G1'
    elif grade == 'B':
        return 5.0, 'G2'
    elif grade == 'C':
        return 5.0, 'G3'
    elif grade == 'L':
        return 5.0, 'オープン(L)'
    elif grade in ['D', 'F', 'G', 'H']:
        return 5.0, '重賞'

    if cond in ['701']:
        return 1.0, '新馬'
    elif cond in ['702', '703']:
        return 1.0, '未勝利'
    elif cond in ['003', '004', '005']:
        return 2.0, '1勝クラス'
    elif cond in ['007', '008', '009', '010']:
        return 3.0, '2勝クラス'
    elif cond in ['014', '015', '016']:
        return 4.0, '3勝クラス'
    elif cond in ['000', '999']:
        return 5.0, 'オープン'

    if is_local:
        return 0.5, '地方'

    return 3.0, '一般'

@st.cache_data(show_spinner=False)
def load_past_races_for_horses(horse_id_list, current_race_date, db_path='C:/Ugaura/sqlite/jra_race.db'):
    horse_id_list = list(horse_id_list)
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

    unique_past_ids = df['past_race_id'].unique().tolist()
    past_placeholders = ','.join(['?'] * len(unique_past_ids))

    # 1. 過去レース全馬走破タイム（UCVトリム平均および勝馬タイム）
    query_all_times = f"""
    SELECT 
        (rd.year || rd.month_day || rd.course_code || rd.times || rd.day || rd.race_number) AS past_race_id,
        hri.blood_reg_number AS horse_id,
        hri.final_order,
        hri.running_time
    FROM race_detail rd
    JOIN horse_race_info hri 
        ON rd.year = hri.year AND rd.month_day = hri.month_day AND rd.course_code = hri.course_code 
        AND rd.times = hri.times AND rd.day = hri.day AND rd.race_number = hri.race_number
    WHERE (rd.year || rd.month_day || rd.course_code || rd.times || rd.day || rd.race_number) IN ({past_placeholders})
      AND hri.abnormality_code IN ('0', '7')
      AND hri.horse_number != '00'
      AND hri.running_time != '0000';
    """
    all_times_df = pd.read_sql(query_all_times, conn, params=list(unique_past_ids))

    race_time_stats = {}
    if not all_times_df.empty:
        for p_id, p_group in all_times_df.groupby('past_race_id'):
            times = []
            winner_t = None
            for _, r in p_group.iterrows():
                t_sec = parse_running_time(r['running_time'])
                if t_sec is not None:
                    times.append(t_sec)
                    order_str = str(r['final_order']).strip()
                    if order_str.isdigit() and int(order_str) == 1:
                        if winner_t is None or t_sec < winner_t:
                            winner_t = t_sec
            if times:
                if winner_t is None:
                    winner_t = min(times)
                count = len(times)
                if count >= 5:
                    t_base = float(trim_mean(times, 0.10))
                else:
                    t_base = float(np.mean(times))
                race_time_stats[str(p_id)] = {'winner_time': winner_t, 'base_time': t_base}

    # 2. 過去走レース出走各馬の直前走（前走）成績集計
    query_pre_races = f"""
    WITH target_past_races AS (
        SELECT DISTINCT 
            (rd.year || rd.month_day || rd.course_code || rd.times || rd.day || rd.race_number) AS past_race_id,
            rd.year || '-' || substr(rd.month_day, 1, 2) || '-' || substr(rd.month_day, 3, 2) AS past_race_date,
            rd.cond_code_youngest AS past_cond,
            rd.grade_code AS past_grade,
            rd.course_code AS past_course
        FROM race_detail rd
        WHERE (rd.year || rd.month_day || rd.course_code || rd.times || rd.day || rd.race_number) IN ({past_placeholders})
    ),
    past_race_horses AS (
        SELECT 
            tpr.past_race_id,
            tpr.past_race_date,
            tpr.past_cond,
            tpr.past_grade,
            tpr.past_course,
            hri.blood_reg_number AS horse_id
        FROM target_past_races tpr
        JOIN horse_race_info hri 
            ON tpr.past_race_id = (hri.year || hri.month_day || hri.course_code || hri.times || hri.day || hri.race_number)
        WHERE hri.abnormality_code IN ('0', '7')
          AND hri.horse_number != '00'
    ),
    horses_prev_races AS (
        SELECT 
            prh.past_race_id,
            prh.horse_id,
            prh.past_cond,
            prh.past_grade,
            prh.past_course,
            hri.final_order AS prev_order,
            rd.cond_code_youngest AS prev_cond,
            rd.grade_code AS prev_grade,
            rd.course_code AS prev_course,
            ROW_NUMBER() OVER (
                PARTITION BY prh.past_race_id, prh.horse_id
                ORDER BY rd.year DESC, rd.month_day DESC, rd.race_number DESC
            ) AS rn
        FROM past_race_horses prh
        JOIN horse_race_info hri ON prh.horse_id = hri.blood_reg_number
        JOIN race_detail rd 
            ON rd.year = hri.year AND rd.month_day = hri.month_day AND rd.course_code = hri.course_code 
            AND rd.times = hri.times AND rd.day = hri.day AND rd.race_number = hri.race_number
        WHERE (rd.year || '-' || substr(rd.month_day, 1, 2) || '-' || substr(rd.month_day, 3, 2)) < prh.past_race_date
          AND hri.abnormality_code IN ('0', '7')
          AND hri.running_time != '0000'
    )
    SELECT 
        past_race_id,
        horse_id,
        past_cond,
        past_grade,
        past_course,
        prev_order,
        prev_cond,
        prev_grade,
        prev_course
    FROM horses_prev_races
    WHERE rn = 1;
    """
    pre_df = pd.read_sql(query_pre_races, conn, params=list(unique_past_ids))

    pre_stats_map = {}
    if not pre_df.empty:
        for past_id, group in pre_df.groupby('past_race_id'):
            same_2nd = 0
            same_3rd = 0
            lower_1st = 0
            for _, r in group.iterrows():
                curr_rank, _ = get_class_info(r['past_cond'], r['past_grade'], r['past_course'])
                prev_rank, _ = get_class_info(r['prev_cond'], r['prev_grade'], r['prev_course'])
                
                p_order_str = str(r['prev_order']).strip()
                if p_order_str.isdigit():
                    p_order = int(p_order_str)
                    if prev_rank == curr_rank:
                        if p_order == 2:
                            same_2nd += 1
                        elif p_order == 3:
                            same_3rd += 1
                    elif prev_rank < curr_rank and p_order == 1:
                        lower_1st += 1
                        
            pl_score = (same_2nd * 1) + (same_3rd * 1) + (lower_1st * 1)
            pre_stats_map[str(past_id)] = {
                'pl_score': pl_score,
                'same_2nd': same_2nd,
                'same_3rd': same_3rd,
                'lower_1st': lower_1st,
                'pre_breakdown_str': f"({same_2nd}-{same_3rd}-{lower_1st})"
            }

    # 3. 過去走レース出走各馬の次走成績集計
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
        CASE WHEN CAST(final_order AS INTEGER) = 1 THEN 1 ELSE 0 END AS self_next_1st,
        CASE WHEN CAST(final_order AS INTEGER) = 2 THEN 1 ELSE 0 END AS self_next_2nd,
        CASE WHEN CAST(final_order AS INTEGER) = 3 THEN 1 ELSE 0 END AS self_next_3rd
    FROM next_races
    WHERE rn = 1;
    """
    
    next_df = pd.read_sql(query_next_races, conn, params=list(unique_past_ids) + [current_race_date])

    query_winner_next = f"""
    WITH target_past_races AS (
        SELECT DISTINCT 
            (rd.year || rd.month_day || rd.course_code || rd.times || rd.day || rd.race_number) AS past_race_id,
            rd.year || '-' || substr(rd.month_day, 1, 2) || '-' || substr(rd.month_day, 3, 2) AS past_race_date
        FROM race_detail rd
        WHERE (rd.year || rd.month_day || rd.course_code || rd.times || rd.day || rd.race_number) IN ({past_placeholders})
    ),
    past_winners AS (
        SELECT 
            tpr.past_race_id,
            tpr.past_race_date,
            hri.blood_reg_number AS winner_horse_id
        FROM target_past_races tpr
        JOIN horse_race_info hri 
            ON tpr.past_race_id = (hri.year || hri.month_day || hri.course_code || hri.times || hri.day || hri.race_number)
        WHERE hri.abnormality_code IN ('0', '7')
          AND CAST(hri.final_order AS INTEGER) = 1
    ),
    winner_next_races AS (
        SELECT 
            pw.past_race_id,
            pw.winner_horse_id,
            hri.final_order AS winner_next_order,
            ROW_NUMBER() OVER (
                PARTITION BY pw.past_race_id, pw.winner_horse_id
                ORDER BY rd.year ASC, rd.month_day ASC, rd.race_number ASC
            ) AS rn
        FROM past_winners pw
        JOIN horse_race_info hri ON pw.winner_horse_id = hri.blood_reg_number
        JOIN race_detail rd 
            ON rd.year = hri.year AND rd.month_day = hri.month_day AND rd.course_code = hri.course_code 
            AND rd.times = hri.times AND rd.day = hri.day AND rd.race_number = hri.race_number
        WHERE (rd.year || '-' || substr(rd.month_day, 1, 2) || '-' || substr(rd.month_day, 3, 2)) > pw.past_race_date
          AND (rd.year || '-' || substr(rd.month_day, 1, 2) || '-' || substr(rd.month_day, 3, 2)) < ?
          AND hri.abnormality_code IN ('0', '7')
          AND hri.running_time != '0000'
    )
    SELECT 
        past_race_id,
        winner_horse_id,
        winner_next_order
    FROM winner_next_races
    WHERE rn = 1;
    """

    winner_next_df = pd.read_sql(query_winner_next, conn, params=list(unique_past_ids) + [current_race_date])
    conn.close()
    winner_next_map = {}
    if not winner_next_df.empty:
        for _, w_row in winner_next_df.iterrows():
            p_id = str(w_row['past_race_id'])
            order_str = str(w_row['winner_next_order']).strip()
            if order_str.isdigit():
                o_int = int(order_str)
                if 1 <= o_int <= 3:
                    winner_next_map[p_id] = str(o_int)
                else:
                    winner_next_map[p_id] = '*'
            else:
                winner_next_map[p_id] = '*'

    df['winner_next_order_code'] = df['past_race_id'].astype(str).map(lambda x: winner_next_map.get(x, '*'))

    winner_pts_map = {'1': 5, '2': 3, '3': 2}

    total_by_race = {}
    if not next_df.empty:
        total_by_race = next_df.groupby('past_race_id').agg(
            total_1st=('self_next_1st', 'sum'),
            total_2nd=('self_next_2nd', 'sum'),
            total_3rd=('self_next_3rd', 'sum'),
            total_next_ran=('self_next_1st', 'count')
        ).to_dict(orient='index')

    other_1st_counts = []
    other_2nd_counts = []
    other_3rd_counts = []
    other_top3_counts = []
    other_next_rans = []
    race_level_scores = []
    race_level_ranks = []
    time_diff_strs = []
    perf_scores = []
    pl_scores = []
    pre_breakdowns = []
    next_breakdowns = []

    for _, row in df.iterrows():
        p_id = str(row['past_race_id'])
        w_code = str(row['winner_next_order_code'])
        
        race_stats = total_by_race.get(p_id, {'total_1st': 0, 'total_2nd': 0, 'total_3rd': 0, 'total_next_ran': 0})
        total_1st = race_stats['total_1st']
        total_2nd = race_stats['total_2nd']
        total_3rd = race_stats['total_3rd']
        total_ran = race_stats['total_next_ran']
        
        w_pts = winner_pts_map.get(w_code, 0)
        l_score = w_pts + (total_1st * 3) + ((total_2nd + total_3rd) * 1)
        
        if l_score >= 10:
            rank = 'S'
        elif l_score >= 6:
            rank = 'A'
        elif l_score >= 3:
            rank = 'B'
        else:
            rank = 'C'

        # タイム差および実力P値の算出
        t_stats = race_time_stats.get(p_id, {})
        winner_t = t_stats.get('winner_time', None)
        self_t = parse_running_time(row.get('race_time', ''))
        
        if self_t is not None and winner_t is not None:
            diff_sec = self_t - winner_t
            if abs(diff_sec) < 0.05:
                td_str = "±0.0"
            elif diff_sec > 0:
                td_str = f"+{diff_sec:.1f}"
            else:
                td_str = f"{diff_sec:.1f}"
            
            k = max(0.0, min(1.0, 1.0 - diff_sec))
            p_score = round(l_score * k, 1)
        else:
            td_str = ""
            p_score = 0.0

        # 前走着度数
        pre_info = pre_stats_map.get(p_id, {'pl_score': 0, 'same_2nd': 0, 'same_3rd': 0, 'lower_1st': 0, 'pre_breakdown_str': '(0-0-0)'})
        pl_scores.append(pre_info['pl_score'])
        pre_breakdowns.append(pre_info['pre_breakdown_str'])

        # 次走着度数: (1着馬の次走着順-次走2着数-次走3着数-次走頭数)
        next_bd_str = f"({w_code}-{total_1st}-{total_2nd}-{total_3rd}-{total_ran})"
        next_breakdowns.append(next_bd_str)

        other_1st_counts.append(total_1st)
        other_2nd_counts.append(total_2nd)
        other_3rd_counts.append(total_3rd)
        other_top3_counts.append(total_1st + total_2nd + total_3rd)
        other_next_rans.append(total_ran)
        race_level_scores.append(l_score)
        race_level_ranks.append(rank)
        time_diff_strs.append(td_str)
        perf_scores.append(p_score)

    df['other_next_1st_count'] = other_1st_counts
    df['other_next_2nd_count'] = other_2nd_counts
    df['other_next_3rd_count'] = other_3rd_counts
    df['other_next_top3_count'] = other_top3_counts
    df['other_next_ran'] = other_next_rans
    df['race_level_score'] = race_level_scores
    df['race_level_rank'] = race_level_ranks
    df['winner_time_diff_str'] = time_diff_strs
    df['perf_score'] = perf_scores
    df['pl_score'] = pl_scores
    df['pre_breakdown_str'] = pre_breakdowns
    df['next_breakdown_str'] = next_breakdowns

    return df

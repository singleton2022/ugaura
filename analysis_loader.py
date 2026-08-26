import sqlite3
import pandas as pd
import streamlit as st

COURSE_MAP = {
    '01': '札幌', '02': '函館', '03': '福島', '04': '新潟', '05': '東京',
    '06': '中山', '07': '中京', '08': '京都', '09': '阪神', '10': '小倉'
}

def format_weight_text(w_val, w_sign_code, w_chg_raw):
    w_val_str = str(w_val).strip() if pd.notna(w_val) else ''
    if not w_val_str.isdigit() or int(w_val_str) == 0:
        return ""
    
    w_base = str(int(w_val_str))
    w_chg_str = str(w_chg_raw).strip() if pd.notna(w_chg_raw) else ''
    
    if not w_chg_str.isdigit():
        return f"{w_base}(±0)"
        
    chg_num = int(w_chg_str)
    if chg_num == 0:
        return f"{w_base}(±0)"
        
    sign_str = str(w_sign_code).strip()
    if sign_str in ['-', '2']:
        return f"{w_base}(-{chg_num})"
    elif sign_str in ['+', '1']:
        return f"{w_base}(+{chg_num})"
    else:
        return f"{w_base}(+{chg_num})"

@st.cache_data(show_spinner=False)
def load_horse_analysis_stats(horse_id_list, current_race_date, target_course_code, target_track_code, target_distance, db_path='C:/Ugaura/sqlite/jra_race.db'):
    horse_id_list = tuple(horse_id_list)
    if not horse_id_list:
        return {}

    conn = sqlite3.connect(db_path)
    placeholders = ','.join(['?'] * len(horse_id_list))

    query = f'''
    SELECT 
        hri.blood_reg_number AS horse_id,
        rd.course_code,
        rd.track_code,
        rd.distance,
        rd.month_day,
        rd.race_sign_code,
        rd.race_name_main,
        hri.bracket_number,
        hri.horse_weight,
        hri.weight_change_sign,
        hri.weight_change,
        hri.final_order,
        ROW_NUMBER() OVER (
            PARTITION BY hri.blood_reg_number 
            ORDER BY rd.year DESC, rd.month_day DESC
        ) AS rn
    FROM horse_race_info hri
    JOIN race_detail rd 
        ON rd.year = hri.year AND rd.month_day = hri.month_day AND rd.course_code = hri.course_code 
        AND rd.times = hri.times AND rd.day = hri.day AND rd.race_number = hri.race_number
    WHERE hri.blood_reg_number IN ({placeholders})
      AND (rd.year || '-' || substr(rd.month_day, 1, 2) || '-' || substr(rd.month_day, 3, 2)) < ?
      AND hri.abnormality_code IN ('0', '7')
      AND hri.horse_number != '00'
      AND hri.running_time != '0000';
    '''

    params = list(horse_id_list) + [current_race_date]
    df = pd.read_sql(query, conn, params=params)
    conn.close()

    target_course = str(target_course_code).zfill(2) if target_course_code is not None else ''
    c_name = COURSE_MAP.get(target_course, '当場')

    try:
        t_code_num = int(target_track_code)
        target_is_turf = 10 <= t_code_num <= 22
    except (ValueError, TypeError):
        target_is_turf = False

    t_short = '芝' if target_is_turf else 'ダ'

    try:
        target_dist_num = int(target_distance)
    except (ValueError, TypeError):
        target_dist_num = 0

    try:
        cur_month = int(current_race_date.split('-')[1])
    except Exception:
        cur_month = 1

    prev_month = (cur_month - 2) % 12 + 1
    next_month = cur_month % 12 + 1
    seasonal_months = {prev_month, cur_month, next_month}

    # 坂コース（中山:06, 中京:07, 阪神:09）
    slope_courses = {'06', '07', '09'}

    # 回り区分
    left_courses = {'04', '05', '07'}
    right_courses = {'01', '02', '03', '06', '08', '09', '10'}
    target_is_left = target_course in left_courses

    result = {}
    for h_id in horse_id_list:
        result[h_id] = {
            'all_track_stats_str': f'全場{t_short}(0-0-0-0)',
            'target_course_stats_str': f'{c_name}{t_short}(0-0-0-0)',
            'course_stats_str': '(0-0-0-0)',
            'course_highlight': False,
            'course_color_type': 'none',
            'right_stats_str': '右(0-0-0-0)',
            'left_stats_str': '左(0-0-0-0)',
            'rotation_color_type': 'none',
            'season_3m_stats_str': f'{prev_month}～{next_month}月(0-0-0-0)',
            'single_month_stats_str': f'{cur_month}月(0-0-0-0)',
            'season_stats_str': '(0-0-0-0)',
            'season_highlight': False,
            'slope_stats_str': '(0-0-0-0)',
            'slope_highlight': False,
            'prev_bracket': '',
            'prev_bracket_num': None,
            'prev_weight_str': '',
            'prev_female_limit': '',
            'prev_is_female_limit': False,
            'first_turf_dirt': '',
            'is_first_turf_dirt': False,
            'distance_reduction': '',
            'is_distance_reduction': False,
        }

    if df.empty:
        return result

    for h_id, group in df.groupby('horse_id'):
        at_1st, at_2nd, at_3rd, at_4th = 0, 0, 0, 0
        c_1st, c_2nd, c_3rd, c_4th = 0, 0, 0, 0
        r_1st, r_2nd, r_3rd, r_4th = 0, 0, 0, 0
        l_1st, l_2nd, l_3rd, l_4th = 0, 0, 0, 0
        s_1st, s_2nd, s_3rd, s_4th = 0, 0, 0, 0
        sm_1st, sm_2nd, sm_3rd, sm_4th = 0, 0, 0, 0
        sl_1st, sl_2nd, sl_3rd, sl_4th = 0, 0, 0, 0
        
        turf_runs = 0
        dirt_runs = 0

        prev_row = None

        for _, row in group.iterrows():
            rn = int(row['rn'])
            if rn == 1:
                prev_row = row

            order_str = str(row['final_order']).strip()
            if not order_str.isdigit():
                continue
            order = int(order_str)

            p_track = row['track_code']
            try:
                p_t_code = int(p_track)
                p_is_turf = 10 <= p_t_code <= 22
            except (ValueError, TypeError):
                p_is_turf = False

            if p_is_turf:
                turf_runs += 1
            else:
                dirt_runs += 1

            p_course = str(row['course_code']).zfill(2)

            # 1-a. 全場芝/ダ成績（今回トラック種別）
            if p_is_turf == target_is_turf:
                if order == 1: at_1st += 1
                elif order == 2: at_2nd += 1
                elif order == 3: at_3rd += 1
                else: at_4th += 1

            # 1-b. 当該競馬場成績（今回コース・トラック種別）
            if p_course == target_course and p_is_turf == target_is_turf:
                if order == 1: c_1st += 1
                elif order == 2: c_2nd += 1
                elif order == 3: c_3rd += 1
                else: c_4th += 1

            # 2. 右回り・左回り集計 (芝・ダート合算)
            if p_course in right_courses:
                if order == 1: r_1st += 1
                elif order == 2: r_2nd += 1
                elif order == 3: r_3rd += 1
                else: r_4th += 1
            elif p_course in left_courses:
                if order == 1: l_1st += 1
                elif order == 2: l_2nd += 1
                elif order == 3: l_3rd += 1
                else: l_4th += 1

            # 3. 月別集計
            m_day = str(row['month_day']).zfill(4)
            p_month = int(m_day[:2])
            if p_month in seasonal_months:
                if order == 1: s_1st += 1
                elif order == 2: s_2nd += 1
                elif order == 3: s_3rd += 1
                else: s_4th += 1
            if p_month == cur_month:
                if order == 1: sm_1st += 1
                elif order == 2: sm_2nd += 1
                elif order == 3: sm_3rd += 1
                else: sm_4th += 1

            # 4. 坂コース集計 (中山06, 中京07, 阪神09 × 同トラック)
            if p_course in slope_courses and p_is_turf == target_is_turf:
                if order == 1: sl_1st += 1
                elif order == 2: sl_2nd += 1
                elif order == 3: sl_3rd += 1
                else: sl_4th += 1

        # 1. 競馬場別成績
        all_track_str = f"全場{t_short}({at_1st}-{at_2nd}-{at_3rd}-{at_4th})"
        target_course_str = f"{c_name}{t_short}({c_1st}-{c_2nd}-{c_3rd}-{c_4th})"

        c_total = c_1st + c_2nd + c_3rd + c_4th
        c_top3 = c_1st + c_2nd + c_3rd
        if c_total > 0:
            c_rate = c_top3 / c_total
            if c_rate >= 0.999:
                c_color = 'red'
                c_hl = True
            elif c_rate >= 0.5:
                c_color = 'yellow'
                c_hl = True
            else:
                c_color = 'none'
                c_hl = False
        else:
            c_color = 'none'
            c_hl = False

        # 2. 右回り・左回り結果
        right_str = f"右({r_1st}-{r_2nd}-{r_3rd}-{r_4th})"
        left_str = f"左({l_1st}-{l_2nd}-{l_3rd}-{l_4th})"

        if target_is_left:
            tgt_total = l_1st + l_2nd + l_3rd + l_4th
            tgt_top3 = l_1st + l_2nd + l_3rd
        else:
            tgt_total = r_1st + r_2nd + r_3rd + r_4th
            tgt_top3 = r_1st + r_2nd + r_3rd

        if tgt_total > 0:
            tgt_rate = tgt_top3 / tgt_total
            if tgt_rate >= 0.999:
                rot_color = 'red'
            elif tgt_rate >= 0.5:
                rot_color = 'yellow'
            else:
                rot_color = 'none'
        else:
            rot_color = 'none'

        # 3. 月別成績
        season_3m_str = f"{prev_month}～{next_month}月({s_1st}-{s_2nd}-{s_3rd}-{s_4th})"
        single_month_str = f"{cur_month}月({sm_1st}-{sm_2nd}-{sm_3rd}-{sm_4th})"

        s_total = s_1st + s_2nd + s_3rd + s_4th
        s_hl = (s_1st + s_2nd + s_3rd) / s_total >= 0.5 if s_total > 0 else False

        # 4. 坂コース結果
        sl_total = sl_1st + sl_2nd + sl_3rd + sl_4th
        sl_str = f"({sl_1st}-{sl_2nd}-{sl_3rd}-{sl_4th})"
        sl_hl = (sl_1st + sl_2nd + sl_3rd) / sl_total >= 0.5 if sl_total > 0 else False

        # 前走枠順
        prev_bracket_str = ''
        prev_brk_num = None
        if prev_row is not None and pd.notna(prev_row.get('bracket_number')):
            raw_brk = str(prev_row['bracket_number']).strip()
            if raw_brk.isdigit() and int(raw_brk) > 0:
                prev_brk_num = int(raw_brk)
                prev_bracket_str = f"{prev_brk_num}枠"

        # 前走馬体重
        prev_weight_str = ''
        if prev_row is not None:
            prev_weight_str = format_weight_text(
                prev_row.get('horse_weight', ''),
                prev_row.get('weight_change_sign', ''),
                prev_row.get('weight_change', '')
            )

        # 前走牝馬限定戦判定
        prev_is_fem = False
        prev_fem_str = ''
        if prev_row is not None:
            p_sign = str(prev_row.get('race_sign_code', '')).strip()
            p_mname = str(prev_row.get('race_name_main', '')).strip()
            if (len(p_sign) >= 2 and p_sign[1] == '2') or ('牝' in p_mname):
                prev_is_fem = True
                prev_fem_str = '牝限'

        # 初芝・初ダート判定
        first_td_str = ''
        is_first_td = False
        if target_is_turf and turf_runs == 0:
            first_td_str = '初芝'
            is_first_td = True
        elif not target_is_turf and dirt_runs == 0:
            first_td_str = '初ダ'
            is_first_td = True

        # 距離短縮（前走より400m以上短縮のみ表示）
        dist_red_str = ''
        is_dist_red = False
        if prev_row is not None and pd.notna(prev_row.get('distance')) and target_dist_num > 0:
            try:
                p_dist_num = int(prev_row['distance'])
                diff_dist = p_dist_num - target_dist_num
                if diff_dist >= 400:
                    dist_red_str = f"-{diff_dist}m"
                    is_dist_red = True
            except (ValueError, TypeError):
                pass

        result[h_id] = {
            'all_track_stats_str': all_track_str,
            'target_course_stats_str': target_course_str,
            'course_stats_str': target_course_str,
            'course_highlight': c_hl,
            'course_color_type': c_color,
            'right_stats_str': right_str,
            'left_stats_str': left_str,
            'rotation_color_type': rot_color,
            'season_3m_stats_str': season_3m_str,
            'single_month_stats_str': single_month_str,
            'season_stats_str': season_3m_str,
            'season_highlight': s_hl,
            'slope_stats_str': sl_str,
            'slope_highlight': sl_hl,
            'prev_bracket': prev_bracket_str,
            'prev_bracket_num': prev_brk_num,
            'prev_weight_str': prev_weight_str,
            'prev_female_limit': prev_fem_str,
            'prev_is_female_limit': prev_is_fem,
            'first_turf_dirt': first_td_str,
            'is_first_turf_dirt': is_first_td,
            'distance_reduction': dist_red_str,
            'is_distance_reduction': is_dist_red,
        }

    return result

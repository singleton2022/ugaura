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

def get_rank_8(score):
    if score is None:
        return 'C'
    if score >= 72.0:
        return 'S'
    elif score >= 66.0:
        return 'A+'
    elif score >= 60.0:
        return 'A'
    elif score >= 54.0:
        return 'B+'
    elif score >= 48.0:
        return 'B'
    elif score >= 42.0:
        return 'C+'
    elif score >= 36.0:
        return 'C'
    else:
        return 'D'

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
    # Cache version: 2026-09-29 07:56 (Fix local race score leak & reset variables)
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
            hri.time_diff,
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
            top5_times = []
            winner_t = None
            for _, r in p_group.iterrows():
                t_sec = parse_running_time(r['running_time'])
                if t_sec is not None:
                    order_str = str(r['final_order']).strip()
                    if order_str.isdigit():
                        ord_i = int(order_str)
                        if ord_i == 1:
                            if winner_t is None or t_sec < winner_t:
                                winner_t = t_sec
                        if 1 <= ord_i <= 5:
                            top5_times.append(t_sec)
            
            top5_avg = float(np.mean(top5_times)) if top5_times else winner_t
            race_time_stats[str(p_id)] = {'top5_avg_time': top5_avg, 'winner_time': winner_t}

    # 2. 新設 race_level テーブルからのレースレベル指標取得
    race_level_map = {}
    try:
        query_race_level = f"""
        SELECT 
            race_id,
            pre_level_score,
            pre_level_rank,
            post_level_score,
            post_level_rank,
            current_level_score,
            current_level_rank,
            next3_total_runs,
            next3_wins,
            next3_2nd,
            next3_3rd,
            winner_next_order
        FROM race_level
        WHERE race_id IN ({past_placeholders});
        """
        rl_df = pd.read_sql(query_race_level, conn, params=list(unique_past_ids))
        if not rl_df.empty:
            race_level_map = rl_df.set_index('race_id').to_dict(orient='index')
    except Exception:
        race_level_map = {}

    conn.close()

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

    rl_pre_scores = []
    rl_pre_ranks = []
    rl_post_scores = []
    rl_post_ranks = []
    rl_current_scores = []
    rl_current_ranks = []
    rl_next3_totals = []
    rl_next3_wins = []
    rl_next3_2nds = []
    rl_next3_3rds = []
    rl_winner_nexts = []
    diff_secs = []
    rl_perf_scores = []
    rl_perf_ranks = []
    rl_perf_adjs = []

    for _, row in df.iterrows():
        p_id = str(row['past_race_id'])
        
        # 毎回のイテレーションで変数を確実に初期化（地方競馬・対象外レースでの持ち越しを完全防止）
        pre_s = None
        pre_r = None
        p_score = None
        p_rank = None
        c_score = None
        c_rank = None
        n3_tot = 0
        n3_w = 0
        n3_2 = 0
        n3_3 = 0
        w_next = None
        l_score = 0
        rank = 'C'
        w_code = '*'
        next_bd_str = "(*-0-0-0-0)"
        total_1st = 0
        total_2nd = 0
        total_3rd = 0
        total_ran = 0

        # 新設 race_level 指標の参照
        rl_info = race_level_map.get(p_id, None)
        if rl_info:
            pre_s = rl_info.get('pre_level_score')
            pre_r = rl_info.get('pre_level_rank')
            p_score = rl_info.get('post_level_score')
            p_rank = rl_info.get('post_level_rank')
            c_score = rl_info.get('current_level_score')
            c_rank = rl_info.get('current_level_rank')
            n3_tot = rl_info.get('next3_total_runs', 0)
            n3_w = rl_info.get('next3_wins', 0)
            n3_2 = rl_info.get('next3_2nd', 0)
            n3_3 = rl_info.get('next3_3rd', 0)
            w_next = rl_info.get('winner_next_order')

            rl_pre_scores.append(pre_s)
            rl_pre_ranks.append(pre_r)
            rl_post_scores.append(p_score)
            rl_post_ranks.append(p_rank)
            rl_current_scores.append(c_score)
            rl_current_ranks.append(c_rank)
            rl_next3_totals.append(n3_tot)
            rl_next3_wins.append(n3_w)
            rl_next3_2nds.append(n3_2)
            rl_next3_3rds.append(n3_3)
            rl_winner_nexts.append(w_next)

            l_score = int(p_score) if p_score is not None else 0
            rank = p_rank if p_rank is not None else 'C'
            w_code = str(w_next) if (w_next is not None and str(w_next).isdigit()) else '*'
            next_bd_str = f"({w_code}-{n3_w}-{n3_2}-{n3_3}-{n3_tot})"
            total_1st = n3_w
            total_2nd = n3_2
            total_3rd = n3_3
            total_ran = n3_tot
        else:
            rl_pre_scores.append(None)
            rl_pre_ranks.append(None)
            rl_post_scores.append(None)
            rl_post_ranks.append(None)
            rl_current_scores.append(None)
            rl_current_ranks.append(None)
            rl_next3_totals.append(0)
            rl_next3_wins.append(0)
            rl_next3_2nds.append(0)
            rl_next3_3rds.append(0)
            rl_winner_nexts.append(None)

            l_score = 0
            rank = 'C'
            w_code = '*'
            next_bd_str = "(*-0-0-0-0)"
            total_1st = 0
            total_2nd = 0
            total_3rd = 0
            total_ran = 0

        # 1〜5着馬平均タイムとの差および実力P値の算出
        order_str = str(row.get('final_order', '')).strip()
        order_int = int(order_str) if order_str.isdigit() else None

        t_stats = race_time_stats.get(p_id, {})
        top5_avg = t_stats.get('top5_avg_time', None)
        winner_t = t_stats.get('winner_time', None)
        self_t = parse_running_time(row.get('race_time', ''))

        td_str = ""
        diff_sec = None

        if self_t is not None and top5_avg is not None:
            # 1〜5着平均タイムとの差: 平均より速い場合はマイナス、遅い場合はプラス
            diff_sec = self_t - top5_avg
            if abs(diff_sec) < 0.05:
                td_str = "±0.0"
            elif diff_sec < 0:
                td_str = f"{diff_sec:.1f}"
            else:
                td_str = f"+{diff_sec:.1f}"
        elif self_t is not None and winner_t is not None:
            # フォールバック (勝ち馬との差)
            diff_sec = self_t - winner_t
            if abs(diff_sec) < 0.05:
                td_str = "±0.0"
            elif diff_sec > 0:
                td_str = f"+{diff_sec:.1f}"
            else:
                td_str = f"{diff_sec:.1f}"

        effective_diff = max(0.0, diff_sec) if diff_sec is not None else 0.0
        k = max(0.0, min(1.0, 1.0 - effective_diff))
        perf_p = round(l_score * k, 1)

        # 自馬パフォーマンス値（perf_score）の算出
        # レースレベル（p_score または c_score）に対して、上位5頭平均差（diff_sec）を加算・減算
        rl_base = p_score if p_score is not None else c_score
        if rl_base is not None and diff_sec is not None:
            # 1秒差 ≒ 約8.0点換算。不利・大敗時の過剰減点を防ぐため adj を [-12.0, +8.0] にクリッピング
            adj = max(-12.0, min(8.0, -(diff_sec * 8.0)))
            p_val = round(max(0.0, min(100.0, rl_base + adj)), 1)
            p_adj = round(adj, 1)
        elif rl_base is not None:
            p_val = round(rl_base, 1)
            p_adj = 0.0
        else:
            p_val = None
            p_adj = 0.0

        p_rnk = get_rank_8(p_val) if p_val is not None else None

        diff_secs.append(diff_sec)
        rl_perf_scores.append(p_val)
        rl_perf_ranks.append(p_rnk)
        rl_perf_adjs.append(p_adj)

        pl_scores.append(0)
        pre_breakdowns.append('(0-0-0)')
        next_breakdowns.append(next_bd_str)
        other_1st_counts.append(total_1st)
        other_2nd_counts.append(total_2nd)
        other_3rd_counts.append(total_3rd)
        other_top3_counts.append(total_1st + total_2nd + total_3rd)
        other_next_rans.append(total_ran)
        race_level_scores.append(l_score)
        race_level_ranks.append(rank)
        time_diff_strs.append(td_str)
        perf_scores.append(perf_p)

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

    df['rl_pre_score'] = rl_pre_scores
    df['rl_pre_rank'] = rl_pre_ranks
    df['rl_post_score'] = rl_post_scores
    df['rl_post_rank'] = rl_post_ranks
    df['rl_current_score'] = rl_current_scores
    df['rl_current_rank'] = rl_current_ranks
    df['rl_next3_total'] = rl_next3_totals
    df['rl_next3_wins'] = rl_next3_wins
    df['rl_next3_2nd'] = rl_next3_2nds
    df['rl_next3_3rd'] = rl_next3_3rds
    df['rl_winner_next'] = rl_winner_nexts
    df['winner_time_diff_sec'] = diff_secs
    df['rl_perf_score'] = rl_perf_scores
    df['rl_perf_rank'] = rl_perf_ranks
    df['rl_perf_adj'] = rl_perf_adjs

    return df

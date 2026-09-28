import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
from fast_loader import get_available_dates, get_races_for_date, load_single_race_card, get_race_level_info
from past_races_loader import load_past_races_for_horses
from analysis_loader import load_horse_analysis_stats

WEEKDAY_JP = ['月', '火', '水', '木', '金', '土', '日']
COURSE_MAP = {
    '01': '札幌', '02': '函館', '03': '福島', '04': '新潟', '05': '東京',
    '06': '中山', '07': '中京', '08': '京都', '09': '阪神', '10': '小倉'
}

def format_date_label(d_str):
    try:
        dt = datetime.strptime(d_str, '%Y-%m-%d')
        w = WEEKDAY_JP[dt.weekday()]
        return f"{d_str} ({w})"
    except Exception:
        return d_str

def calc_interval_info(recent_date_str, past_date_str):
    if not recent_date_str or not past_date_str:
        return " ", "text-align: center;"
    try:
        dt_recent = datetime.strptime(str(recent_date_str).strip(), '%Y-%m-%d')
        dt_past = datetime.strptime(str(past_date_str).strip(), '%Y-%m-%d')
        days = (dt_recent - dt_past).days
        
        if days <= 9:
            return "=", "background-color: #FFF59D; color: #000; font-weight: bold; text-align: center;"
        elif 70 <= days < 182:
            return "+", "background-color: #AED581; color: #000; font-weight: bold; text-align: center;"
        elif days >= 182:
            return "+", "background-color: #EF5350; color: #FFF; font-weight: bold; text-align: center;"
        else:
            return " ", "text-align: center;"
    except Exception:
        return " ", "text-align: center;"

def get_track_dist_label(track_code, distance):
    if pd.isna(distance) or not str(distance).strip().isdigit():
        return "", ""
    d_num = int(distance)
    d_hundred = d_num // 100
    
    try:
        t_code_num = int(track_code)
        is_turf = 10 <= t_code_num <= 22
    except (ValueError, TypeError):
        is_turf = False

    if is_turf:
        return f"芝{d_hundred}", "color: #2e7d32; font-weight: bold; padding: 2px 4px;"
    else:
        return f"ダ{d_hundred}", "color: #795548; font-weight: bold; padding: 2px 4px;"

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
            is_local = True

    if is_local:
        return 0.5, '地方'

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

    return 3.0, '一般'

def format_horse_weight(w_val, w_sign_code, w_chg_raw):
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

def format_win_odds(odds_raw):
    o_str = str(odds_raw).strip() if pd.notna(odds_raw) else ''
    if not o_str.isdigit() or int(o_str) == 0:
        return ""
    val = float(int(o_str)) / 10.0
    return f"{val:.1f}"

def format_weight_carried(wc):
    if pd.isna(wc):
        return ""
    val_str = str(wc).strip()
    if not val_str:
        return ""
    try:
        val_float = float(val_str)
        if val_float > 100:
            val_float = val_float / 10.0
        return f"{val_float:.1f}"
    except ValueError:
        return val_str

def fetch_dates():
    return get_available_dates()

def render_header_and_legend(target_race_df, selected_date, legend_html):
    course_code = str(target_race_df['course_code'].iloc[0]).zfill(2)
    course_name = COURSE_MAP.get(course_code, '競馬場')
    race_num = int(target_race_df['race_number'].iloc[0])
    
    raw_short10 = str(target_race_df['race_name_short_10'].iloc[0]).strip() if pd.notna(target_race_df['race_name_short_10'].iloc[0]) else ''
    raw_main = str(target_race_df['race_name_main'].iloc[0]).strip() if pd.notna(target_race_df['race_name_main'].iloc[0]) else ''
    race_name = raw_short10 if raw_short10 else (raw_main if raw_main else '')
    
    track_code = target_race_df['track_code'].iloc[0]
    distance = target_race_df['distance'].iloc[0]
    try:
        is_turf = 10 <= int(track_code) <= 22
    except Exception:
        is_turf = False
    t_str = '芝' if is_turf else 'ダ'
    dist_str = f"{t_str}{distance}m"
    
    cond_code = target_race_df['cond_code_youngest'].iloc[0]
    grade_code = target_race_df['grade_code'].iloc[0]
    _, class_name = get_class_info(cond_code, grade_code, course_code)
    
    race_sign = str(target_race_df['race_sign_code'].iloc[0]).strip() if 'race_sign_code' in target_race_df.columns else ''
    is_female_limit = (len(race_sign) >= 2 and race_sign[1] == '2') or ('牝' in race_name) or ('牝' in raw_main)
    
    fem_badge = '<span style="background-color: #0284c7; color: #ffffff; font-weight: bold; padding: 2px 6px; border-radius: 4px; font-size: 0.85em; margin-left: 6px;">牝限</span>' if is_female_limit else ''
    
    date_label = format_date_label(selected_date)
    
    if race_name:
        race_title_str = f"{course_name} {race_num}R {race_name}"
    else:
        race_title_str = f"{course_name} {race_num}R"

    # レースレベル情報 (Pre-Score) の取得
    race_id = target_race_df['race_id'].iloc[0] if 'race_id' in target_race_df.columns else None
    pre_badge_html = ""
    if race_id:
        rl_info = get_race_level_info(race_id)
        if rl_info and pd.notna(rl_info.get('pre_level_score')):
            pre_score = rl_info['pre_level_score']
            pre_rank = rl_info.get('pre_level_rank', 'C')
            post_score = rl_info.get('post_level_score')
            post_rank = rl_info.get('post_level_rank')

            rank_badge_styles = {
                'S': 'background-color: #dc2626; color: #ffffff;',   # 濃赤 (G1級・超一線級)
                'A+': 'background-color: #ea580c; color: #ffffff;',  # 鮮やかな赤橙 (重賞上位・ハイレベルOP)
                'A': 'background-color: #d97706; color: #ffffff;',   # アンバー (3勝上位・OP標準・2歳G1)
                'B+': 'background-color: #2563eb; color: #ffffff;',  # 鮮やかなブルー (2勝上位・3勝標準・2歳重賞)
                'B': 'background-color: #10b981; color: #ffffff;',   # エメラルドグリーン (2勝標準・1勝上位)
                'C+': 'background-color: #0d9488; color: #ffffff;',  # ティール青緑 (1勝標準・未勝利上位)
                'C': 'background-color: #64748b; color: #ffffff;',   # スレートグレー (未勝利標準)
                'D': 'background-color: #94a3b8; color: #ffffff;',   # ライトグレー (低調戦)
            }
            b_style = rank_badge_styles.get(pre_rank, 'background-color: #64748b; color: #fff;')
            
            post_note = ""
            if pd.notna(post_score) and rl_info.get('post_level_score') is not None:
                post_b_style = rank_badge_styles.get(post_rank, 'background-color: #64748b; color: #fff;')
                post_note = f' <span style="font-size: 0.85em; color: #94a3b8; margin-left: 6px;">(事後確定: <span style="{post_b_style} font-weight: bold; padding: 1px 5px; border-radius: 3px;">{post_rank} {post_score:.1f}</span>)</span>'
                
            pre_badge_html = f"""<div style="margin-top: 6px; font-size: 0.92em; color: #cbd5e1; display: flex; align-items: center; flex-wrap: wrap;">
                <span>Pre-Score (レース前レベル): </span>
                <span style="{b_style} font-weight: bold; padding: 2px 8px; border-radius: 4px; font-size: 0.95em; margin-left: 6px; box-shadow: 0 1px 2px rgba(0,0,0,0.2);">{pre_rank} {pre_score:.1f}</span>
                {post_note}
            </div>"""
        
    header_html = f"""<div style="display: flex; justify-content: space-between; align-items: stretch; margin-bottom: 12px; gap: 12px; flex-wrap: wrap;">
    <div style="flex: 1; min-width: 320px; padding: 10px 14px; background-color: #1e293b; border-radius: 8px; border: 1px solid #334155; box-shadow: 0 2px 4px rgba(0,0,0,0.12); display: flex; flex-direction: column; justify-content: center;">
        <div style="font-size: 1.15em; font-weight: bold; color: #f8fafc; margin-bottom: 4px;">
            🏇 {date_label} {race_title_str}
        </div>
        <div style="font-size: 0.95em; color: #94a3b8; font-weight: 500;">
            <span style="color: #e2e8f0; font-weight: bold;">{dist_str}</span> ｜ クラス: <span style="font-weight: bold; color: #38bdf8;">{class_name}</span>{fem_badge}
        </div>
        {pre_badge_html}
    </div>
    <div style="flex: 1.2; min-width: 340px; padding: 8px 12px; background-color: #1e293b; border-radius: 8px; border: 1px solid #334155; box-shadow: 0 2px 4px rgba(0,0,0,0.12); font-size: 0.85em; color: #cbd5e1; line-height: 1.5; display: flex; flex-direction: column; justify-content: center;">
        {legend_html}
    </div>
</div>"""

    if hasattr(st, 'html'):
        st.html(header_html)
    else:
        st.markdown(header_html, unsafe_allow_html=True)

def render_standard_card(target_race_df, selected_date, target_rank, target_class_name):
    legend_html = """<div>
        <span style="font-weight: bold; color: #38bdf8;">💡 過去5走（自馬パフォーマンス表示）:</span><br>
        <span style="color: #f1f5f9;">【上段】自馬着順(上位5頭平均差) ｜ <span style="background-color: #dc2626; color: #fff; padding: 1px 4px; border-radius: 3px; font-weight: bold;">S</span>〜<span style="background-color: #94a3b8; color: #fff; padding: 1px 4px; border-radius: 3px; font-weight: bold;">D</span> 自馬パフォーマンス [ランク スコア] <span style="color: #94a3b8; font-size: 0.88em;">(※レースレベルに上位5頭平均差を加算・減算)</span></span><br>
        <span style="color: #f1f5f9;">【下段】出走各馬の次3走成績 (1着数-2着数-3着数 / 走数) [勝馬次走]</span>
    </div>"""
    
    render_header_and_legend(target_race_df, selected_date, legend_html)

    # 今回レースの Pre-Score を取得（前走との相手関係比較用）
    target_race_id = target_race_df['race_id'].iloc[0] if 'race_id' in target_race_df.columns else None
    target_rl = get_race_level_info(target_race_id) if target_race_id else None
    target_pre_score = target_rl.get('pre_level_score') if target_rl else None

    horse_id_list = target_race_df['horse_id'].unique().tolist()
    past_df = load_past_races_for_horses(tuple(horse_id_list), selected_date)

    past_5_map = {}
    if not past_df.empty:
        for h_id, group in past_df.groupby('horse_id'):
            past_5_map[h_id] = {}
            for _, row in group.iterrows():
                rn = int(row['rn'])
                if 1 <= rn <= 5:
                    order = row.get('final_order', '')
                    if pd.notna(order) and str(order).strip().isdigit() and int(order) > 0:
                        self_order_str = f"{int(order)}着"
                    else:
                        self_order_str = ""
                        
                    level_score = row.get('race_level_score', 0)
                    level_rank = row.get('race_level_rank', 'C')
                    time_diff_str = row.get('winner_time_diff_str', '')
                    perf_score = row.get('perf_score', 0.0)

                    td_disp = f"({time_diff_str})" if time_diff_str else ""
                    
                    # 事後確定レースレベル (Post-Score優先、未出走の場合はCurrent-Score)
                    rl_score = row.get('rl_post_score')
                    rl_rank = row.get('rl_post_rank')
                    if pd.isna(rl_score) or rl_score is None:
                        rl_score = row.get('rl_current_score')
                        rl_rank = row.get('rl_current_rank')

                    # 自馬パフォーマンス値 (レースレベルに上位5頭平均差を加算・減算した実質能力値)
                    perf_score = row.get('rl_perf_score')
                    perf_rank = row.get('rl_perf_rank')
                    perf_adj = row.get('rl_perf_adj', 0.0)
                    if pd.isna(perf_score) or perf_score is None:
                        perf_score = rl_score
                        perf_rank = rl_rank

                    # ツールチップ用: 事前水準(Pre-Score)の取得
                    pre_s = row.get('rl_pre_score')
                    pre_r = row.get('rl_pre_rank')

                    rank_badge_styles = {
                        'S': 'background-color: #dc2626; color: #ffffff;',   # 濃赤 (G1級・超一線級)
                        'A+': 'background-color: #ea580c; color: #ffffff;',  # 鮮やかな赤橙 (重賞上位・ハイレベルOP)
                        'A': 'background-color: #d97706; color: #ffffff;',   # アンバー (3勝上位・OP標準・2歳G1)
                        'B+': 'background-color: #2563eb; color: #ffffff;',  # 鮮やかなブルー (2勝上位・3勝標準・2歳重賞)
                        'B': 'background-color: #10b981; color: #ffffff;',   # エメラルドグリーン (2勝標準・1勝上位)
                        'C+': 'background-color: #0d9488; color: #ffffff;',  # ティール青緑 (1勝標準・未勝利上位)
                        'C': 'background-color: #64748b; color: #ffffff;',   # スレートグレー (未勝利標準)
                        'D': 'background-color: #94a3b8; color: #ffffff;',   # ライトグレー (低調戦)
                    }

                    if pd.notna(perf_rank) and perf_rank in rank_badge_styles and pd.notna(perf_score):
                        b_style = rank_badge_styles[perf_rank]
                        badge_title = f"自馬パフォーマンス: {perf_rank} {perf_score:.1f}"
                        if pd.notna(rl_score):
                            badge_title += f" ｜ 元レースレベル: {rl_rank} {rl_score:.1f} (タイム補正: {perf_adj:+.1f}点)"
                        if pd.notna(pre_s):
                            badge_title += f" ｜ レース前水準(Pre): {pre_r} {pre_s:.1f}"
                        if rn == 1 and target_pre_score is not None and pd.notna(pre_s):
                            diff = pre_s - target_pre_score
                            if diff >= 3.0:
                                badge_title += f" ｜ 今回比: {diff:+.1f}点 (相手緩和/前走格上)"
                            elif diff <= -3.0:
                                badge_title += f" ｜ 今回比: {diff:+.1f}点 (相手強化/前走格下)"
                            else:
                                badge_title += f" ｜ 今回比: {diff:+.1f}点 (同水準)"

                        badge_html = f"<span title='{badge_title}' style='{b_style} font-weight: bold; padding: 1px 5px; border-radius: 3px; font-size: 0.85em; margin-left: 4px; cursor: help;'>{perf_rank} {perf_score:.1f}</span>"
                    else:
                        badge_html = ""

                    line1 = f"<span>{self_order_str}{td_disp}</span>{badge_html}".strip()

                    # 下段: 次3走成績
                    n3_total = row.get('rl_next3_total', 0)
                    n3_w = row.get('rl_next3_wins', 0)
                    n3_2 = row.get('rl_next3_2nd', 0)
                    n3_3 = row.get('rl_next3_3rd', 0)
                    w_next = row.get('rl_winner_next')

                    if pd.notna(n3_total) and int(n3_total) > 0:
                        bd_str = f"次3走:{int(n3_w)}-{int(n3_2)}-{int(n3_3)}/{int(n3_total)}走"
                        if pd.notna(w_next):
                            try:
                                w_int = int(float(w_next))
                                if 1 <= w_int <= 3:
                                    w_circle = {1: '①', 2: '②', 3: '③'}[w_int]
                                    bd_str += f" (勝{w_circle})"
                                elif w_int > 0:
                                    bd_str += f" (勝{w_int})"
                            except (ValueError, TypeError):
                                pass
                    else:
                        bd_str = "次3走: 未出走"

                    line2 = f"<span style='font-size: 0.88em; color: #64748b;'>{bd_str}</span>"
                    
                    cell_html = f"<div style='line-height: 1.3;'><div>{line1}</div><div>{line2}</div></div>"
                    
                    p_cond = row.get('cond_code_youngest', None)
                    p_grade = row.get('grade_code', None)
                    p_course = row.get('course_code', None)
                    p_rank, p_cname = get_class_info(p_cond, p_grade, p_course)
                    
                    is_p_local = False
                    if p_course is not None:
                        try:
                            pc_num = int(str(p_course).strip())
                            if pc_num not in range(1, 11):
                                is_p_local = True
                        except (ValueError, TypeError):
                            is_p_local = True

                    if is_p_local or p_cname == '地方' or p_rank == 0.5:
                        comp = 'local'
                    elif p_rank == target_rank:
                        comp = 'same'
                    elif p_rank < target_rank:
                        comp = 'lower'
                    else:
                        comp = 'higher'

                    track_code = row.get('track_code', None)
                    distance = row.get('distance', None)
                    td_label, td_style = get_track_dist_label(track_code, distance)
                    
                    past_5_map[h_id][rn] = {
                        'text': cell_html,
                        'comp': comp,
                        'class_name': p_cname,
                        'race_date': row.get('race_date', ''),
                        'track_dist_label': td_label,
                        'track_dist_style': td_style,
                        'race_level_score': level_score,
                        'race_level_rank': level_rank,
                        'perf_score': perf_score
                    }

    card_rows = []
    comp_matrix = []
    interval_matrix = []
    
    sex_map = {'1': '牡', '2': '牝', '3': 'セ'}

    for idx, row in target_race_df.iterrows():
        h_id = row['horse_id']
        h_num = str(row['horse_number'])
        bracket = str(row['bracket_number']) if pd.notna(row['bracket_number']) else ''
        h_name = row['horse_name']
        
        sex = sex_map.get(str(row['sex_code']), '')
        
        raw_age = row.get('horse_age', '')
        if pd.notna(raw_age) and str(raw_age).strip().isdigit() and int(raw_age) > 0:
            age_str = str(int(raw_age))
        else:
            age_str = ""
            
        sex_age = f"{sex}{age_str}"
        
        weight_carried = format_weight_carried(row.get('weight_carried', ''))
        jockey = row['jockey_name_short'] if pd.notna(row['jockey_name_short']) else ''
        
        odds_fmt = format_win_odds(row.get('win_odds', ''))

        h_weight = format_horse_weight(
            row.get('horse_weight', ''), 
            row.get('weight_change_sign', ''), 
            row.get('weight_change', '')
        )

        p_data = past_5_map.get(h_id, {})
        r1 = p_data.get(1, {'text': '', 'comp': 'none', 'race_date': '', 'track_dist_label': '', 'track_dist_style': ''})
        r2 = p_data.get(2, {'text': '', 'comp': 'none', 'race_date': '', 'track_dist_label': '', 'track_dist_style': ''})
        r3 = p_data.get(3, {'text': '', 'comp': 'none', 'race_date': '', 'track_dist_label': '', 'track_dist_style': ''})
        r4 = p_data.get(4, {'text': '', 'comp': 'none', 'race_date': '', 'track_dist_label': '', 'track_dist_style': ''})
        r5 = p_data.get(5, {'text': '', 'comp': 'none', 'race_date': '', 'track_dist_label': '', 'track_dist_style': ''})

        i0_char, i0_style = calc_interval_info(selected_date, r1.get('race_date', ''))
        i1_char, i1_style = calc_interval_info(r1.get('race_date', ''), r2.get('race_date', ''))
        i2_char, i2_style = calc_interval_info(r2.get('race_date', ''), r3.get('race_date', ''))
        i3_char, i3_style = calc_interval_info(r3.get('race_date', ''), r4.get('race_date', ''))
        i4_char, i4_style = calc_interval_info(r4.get('race_date', ''), r5.get('race_date', ''))

        card_rows.append({
            '馬番': int(h_num),
            '枠': bracket,
            '馬名': h_name,
            '性齢': sex_age,
            '騎手': jockey,
            '斤量': weight_carried,
            'オッズ': odds_fmt,
            '馬体重': h_weight,
            'r1': r1['text'],
            'r1_td_label': r1['track_dist_label'],
            'r1_td_style': r1['track_dist_style'],
            'r2': r2['text'],
            'r2_td_label': r2['track_dist_label'],
            'r2_td_style': r2['track_dist_style'],
            'r3': r3['text'],
            'r3_td_label': r3['track_dist_label'],
            'r3_td_style': r3['track_dist_style'],
            'r4': r4['text'],
            'r4_td_label': r4['track_dist_label'],
            'r4_td_style': r4['track_dist_style'],
            'r5': r5['text'],
            'r5_td_label': r5['track_dist_label'],
            'r5_td_style': r5['track_dist_style'],
            'i0_char': i0_char,
            'i1_char': i1_char,
            'i2_char': i2_char,
            'i3_char': i3_char,
            'i4_char': i4_char,
        })
        
        comp_matrix.append({
            1: r1['comp'],
            2: r2['comp'],
            3: r3['comp'],
            4: r4['comp'],
            5: r5['comp']
        })

        interval_matrix.append({
            0: (i0_char, i0_style),
            1: (i1_char, i1_style),
            2: (i2_char, i2_style),
            3: (i3_char, i3_style),
            4: (i4_char, i4_style),
        })

    disp_df = pd.DataFrame(card_rows).sort_values('馬番').reset_index(drop=True)
    card_rows = disp_df.to_dict(orient='records')

    base_bg_map = {
        'same': 'background-color: #FFF59D;',
        'lower': 'background-color: #81D4FA;',
        'higher': 'background-color: #CE93D8;',
        'local': 'background-color: #E0E0E0;',
        'none': ''
    }

    def build_cell_style(comp):
        bg = base_bg_map.get(comp, '')
        return f"{bg} color: #000; font-weight: bold;" if bg else ""

    html_code = """<style>
.race-card-container {
    width: 100%;
    overflow-x: auto;
    margin-bottom: 20px;
}
.race-card-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}
.race-card-table th, .race-card-table td {
    border: 1px solid #dcdfe6;
    padding: 6px 4px;
    text-align: center;
    vertical-align: middle;
    white-space: nowrap;
}
.race-card-table th {
    background-color: #f2f4f7;
    color: #333;
    font-weight: 600;
}
.race-card-table td.col-int, .race-card-table th.col-int {
    width: 18px !important;
    min-width: 18px !important;
    max-width: 18px !important;
    padding: 2px 0px !important;
    font-size: 12px;
    font-weight: bold;
}
.race-card-table td.horse-name {
    text-align: left;
    padding-left: 8px;
    font-weight: bold;
}
</style>
<div class="race-card-container">
<table class="race-card-table">
<thead>
<tr>
<th>枠</th>
<th>馬番</th>
<th>馬名</th>
<th>性齢</th>
<th>騎手</th>
<th>斤量</th>
<th>オッズ</th>
<th>馬体重</th>
<th class="col-int"></th>
<th colspan="2">前走</th>
<th class="col-int"></th>
<th colspan="2">２走前</th>
<th class="col-int"></th>
<th colspan="2">３走前</th>
<th class="col-int"></th>
<th colspan="2">４走前</th>
<th class="col-int"></th>
<th colspan="2">５走前</th>
</tr>
</thead>
<tbody>"""

    for i, r in enumerate(card_rows):
        comp = comp_matrix[i]
        int_m = interval_matrix[i]
        
        c1_style = build_cell_style(comp[1])
        c2_style = build_cell_style(comp[2])
        c3_style = build_cell_style(comp[3])
        c4_style = build_cell_style(comp[4])
        c5_style = build_cell_style(comp[5])

        i0_char, i0_st = int_m[0]
        i1_char, i1_st = int_m[1]
        i2_char, i2_st = int_m[2]
        i3_char, i3_st = int_m[3]
        i4_char, i4_st = int_m[4]

        html_code += f"""<tr>
<td>{r['枠']}</td>
<td>{r['馬番']}</td>
<td class="horse-name">{r['馬名']}</td>
<td>{r['性齢']}</td>
<td>{r['騎手']}</td>
<td>{r['斤量']}</td>
<td>{r['オッズ']}</td>
<td>{r['馬体重']}</td>
<td class="col-int" style="{i0_st}">{i0_char}</td>
<td style="{r['r1_td_style']}">{r['r1_td_label']}</td>
<td style="{c1_style}">{r['r1']}</td>
<td class="col-int" style="{i1_st}">{i1_char}</td>
<td style="{r['r2_td_style']}">{r['r2_td_label']}</td>
<td style="{c2_style}">{r['r2']}</td>
<td class="col-int" style="{i2_st}">{i2_char}</td>
<td style="{r['r3_td_style']}">{r['r3_td_label']}</td>
<td style="{c3_style}">{r['r3']}</td>
<td class="col-int" style="{i3_st}">{i3_char}</td>
<td style="{r['r4_td_style']}">{r['r4_td_label']}</td>
<td style="{c4_style}">{r['r4']}</td>
<td class="col-int" style="{i4_st}">{i4_char}</td>
<td style="{r['r5_td_style']}">{r['r5_td_label']}</td>
<td style="{c5_style}">{r['r5']}</td>
</tr>"""

    html_code += """</tbody>
</table>
</div>"""

    if hasattr(st, 'html'):
        st.html(html_code)
    else:
        st.markdown(html_code, unsafe_allow_html=True)

def render_analysis_card(target_race_df, selected_date):
    legend_html = """<div>
        <span style="font-weight: bold; color: #38bdf8;">💡 凡例:</span><br>
        <span style="background-color: #ef4444; color: #fff; font-weight: bold; padding: 1px 5px; border-radius: 3px;">赤色</span> 同コース/同回り3着内率100%・前走枠(芝外➔内/ダ内➔外)・400m以上短縮<br>
        <span style="background-color: #0284c7; color: #fff; font-weight: bold; padding: 1px 5px; border-radius: 3px;">水色</span> 前走枠(芝内➔外/ダ外➔内)・前走牝限・初芝ダ<br>
        <span style="background-color: #eab308; color: #000; font-weight: bold; padding: 1px 5px; border-radius: 3px;">黄色</span> 着度数複勝率50%以上
    </div>"""
    
    render_header_and_legend(target_race_df, selected_date, legend_html)

    horse_id_list = target_race_df['horse_id'].unique().tolist()
    target_course = str(target_race_df['course_code'].iloc[0]).zfill(2) if 'course_code' in target_race_df.columns else '01'
    target_track = target_race_df['track_code'].iloc[0] if 'track_code' in target_race_df.columns else '10'
    target_dist = target_race_df['distance'].iloc[0] if 'distance' in target_race_df.columns else 0

    has_slope_col = target_course in ['06', '07', '09']

    stats_map = load_horse_analysis_stats(tuple(horse_id_list), selected_date, target_course, target_track, target_dist)

    sex_map = {'1': '牡', '2': '牝', '3': 'セ'}
    card_rows = []

    for idx, row in target_race_df.iterrows():
        h_id = row['horse_id']
        h_num = str(row['horse_number'])
        bracket = str(row['bracket_number']) if pd.notna(row['bracket_number']) else ''
        h_name = row['horse_name']
        
        sex = sex_map.get(str(row['sex_code']), '')
        raw_age = row.get('horse_age', '')
        if pd.notna(raw_age) and str(raw_age).strip().isdigit() and int(raw_age) > 0:
            age_str = str(int(raw_age))
        else:
            age_str = ""
            
        sex_age = f"{sex}{age_str}"
        weight_carried = format_weight_carried(row.get('weight_carried', ''))
        jockey = row['jockey_name_short'] if pd.notna(row['jockey_name_short']) else ''
        odds_fmt = format_win_odds(row.get('win_odds', ''))
        h_weight = format_horse_weight(
            row.get('horse_weight', ''), 
            row.get('weight_change_sign', ''), 
            row.get('weight_change', '')
        )

        h_stats = stats_map.get(h_id, {
            'all_track_stats_str': '全場(0-0-0-0)',
            'target_course_stats_str': '当場(0-0-0-0)',
            'course_stats_str': '(0-0-0-0)',
            'course_highlight': False,
            'course_color_type': 'none',
            'right_stats_str': '右(0-0-0-0)',
            'left_stats_str': '左(0-0-0-0)',
            'rotation_color_type': 'none',
            'season_3m_stats_str': '3ヶ月(0-0-0-0)',
            'single_month_stats_str': '当月(0-0-0-0)',
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
        })

        c_color_type = h_stats.get('course_color_type', 'none')
        if c_color_type == 'red':
            course_class = "highlight-red"
        elif c_color_type == 'yellow':
            course_class = "highlight-yellow"
        else:
            course_class = ""

        rot_color = h_stats.get('rotation_color_type', 'none')
        if rot_color == 'red':
            rot_class = "highlight-red"
        elif rot_color == 'yellow':
            rot_class = "highlight-yellow"
        else:
            rot_class = ""

        course_html = f"<div>{h_stats['all_track_stats_str']}</div><div>{h_stats['target_course_stats_str']}</div>"
        rot_html = f"<div>{h_stats['right_stats_str']}</div><div>{h_stats['left_stats_str']}</div>"
        season_html = f"<div>{h_stats['season_3m_stats_str']}</div><div>{h_stats['single_month_stats_str']}</div>"

        cur_brk_num = int(bracket) if bracket.isdigit() else None
        prev_brk_num = h_stats.get('prev_bracket_num', None)
        bracket_color_class = ""
        if cur_brk_num is not None and prev_brk_num is not None:
            track_num = int(target_track) if str(target_track).isdigit() else 10
            is_turf = 10 <= track_num <= 22
            if is_turf:
                if 1 <= prev_brk_num <= 3 and 6 <= cur_brk_num <= 8:
                    bracket_color_class = "highlight-cyan"
                elif 6 <= prev_brk_num <= 8 and 1 <= cur_brk_num <= 3:
                    bracket_color_class = "highlight-red"
            else:
                if 6 <= prev_brk_num <= 8 and 1 <= cur_brk_num <= 3:
                    bracket_color_class = "highlight-cyan"
                elif 1 <= prev_brk_num <= 3 and 6 <= cur_brk_num <= 8:
                    bracket_color_class = "highlight-red"

        card_rows.append({
            '馬番': int(h_num),
            '枠': bracket,
            '馬名': h_name,
            '性齢': sex_age,
            '騎手': jockey,
            '斤量': weight_carried,
            'オッズ': odds_fmt,
            '馬体重': h_weight,
            'course_html': course_html,
            'course_class': course_class,
            'rot_html': rot_html,
            'rot_class': rot_class,
            'season_html': season_html,
            'season_hl': h_stats['season_highlight'],
            'slope_stats': h_stats['slope_stats_str'],
            'slope_hl': h_stats['slope_highlight'],
            'prev_bracket': h_stats['prev_bracket'],
            'bracket_color_class': bracket_color_class,
            'prev_weight': h_stats['prev_weight_str'],
            'prev_female_limit': h_stats['prev_female_limit'],
            'female_hl': h_stats['prev_is_female_limit'],
            'first_turf_dirt': h_stats['first_turf_dirt'],
            'first_td_hl': h_stats['is_first_turf_dirt'],
            'dist_reduction': h_stats['distance_reduction'],
            'dist_red_hl': h_stats['is_distance_reduction'],
        })

    disp_df = pd.DataFrame(card_rows).sort_values('馬番').reset_index(drop=True)
    card_rows = disp_df.to_dict(orient='records')

    html_code = """<style>
.race-card-container {
    width: 100%;
    overflow-x: auto;
    margin-bottom: 20px;
}
.race-card-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}
.race-card-table th, .race-card-table td {
    border: 1px solid #dcdfe6;
    padding: 6px 8px;
    text-align: center;
    vertical-align: middle;
    white-space: nowrap;
}
.race-card-table th {
    background-color: #f2f4f7;
    color: #333;
    font-weight: 600;
}
.race-card-table td.horse-name {
    text-align: left;
    padding-left: 8px;
    font-weight: bold;
}
.highlight-yellow {
    background-color: #FFF59D !important;
    color: #000 !important;
    font-weight: bold !important;
}
.highlight-cyan {
    background-color: #81D4FA !important;
    color: #000 !important;
    font-weight: bold !important;
}
.highlight-red {
    background-color: #FFCDD2 !important;
    color: #B71C1C !important;
    font-weight: bold !important;
}
</style>
<div class="race-card-container">
<table class="race-card-table">
<thead>
<tr>
<th>枠</th>
<th>馬番</th>
<th>馬名</th>
<th>性齢</th>
<th>騎手</th>
<th>斤量</th>
<th>オッズ</th>
<th>馬体重</th>
<th>競馬場別成績</th>
<th>回り別成績</th>
<th>月別成績</th>"""

    if has_slope_col:
        html_code += """
<th>坂コース成績</th>"""

    html_code += """
<th>前走枠</th>
<th>前走馬体重</th>
<th>前走牝限</th>
<th>初芝/ダ</th>
<th>距離短縮</th>
</tr>
</thead>
<tbody>"""

    for r in card_rows:
        c_class = r['course_class']
        rot_class = r['rot_class']
        s_class = "highlight-yellow" if r['season_hl'] else ""
        sl_class = "highlight-yellow" if r['slope_hl'] else ""
        brk_class = r['bracket_color_class']
        fem_class = "highlight-cyan" if r['female_hl'] else ""
        ftd_class = "highlight-cyan" if r['first_td_hl'] else ""
        dred_class = "highlight-red" if r['dist_red_hl'] else ""

        html_code += f"""<tr>
<td>{r['枠']}</td>
<td>{r['馬番']}</td>
<td class="horse-name">{r['馬名']}</td>
<td>{r['性齢']}</td>
<td>{r['騎手']}</td>
<td>{r['斤量']}</td>
<td>{r['オッズ']}</td>
<td>{r['馬体重']}</td>
<td class="{c_class}">{r['course_html']}</td>
<td class="{rot_class}">{r['rot_html']}</td>
<td class="{s_class}">{r['season_html']}</td>"""

        if has_slope_col:
            html_code += f"""
<td class="{sl_class}">{r['slope_stats']}</td>"""

        html_code += f"""
<td class="{brk_class}">{r['prev_bracket']}</td>
<td>{r['prev_weight']}</td>
<td class="{fem_class}">{r['prev_female_limit']}</td>
<td class="{ftd_class}">{r['first_turf_dirt']}</td>
<td class="{dred_class}">{r['dist_reduction']}</td>
</tr>"""

    html_code += """</tbody>
</table>
</div>"""

    if hasattr(st, 'html'):
        st.html(html_code)
    else:
        st.markdown(html_code, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="出馬表", layout="wide")

    st.markdown("""
    <style>
        header[data-testid="stHeader"] {
            background-color: transparent !important;
        }
        .block-container {
            padding-top: 2.0rem !important;
            padding-bottom: 0.5rem !important;
        }
        h1 {
            padding-top: 0rem !important;
            margin-top: 0rem !important;
            margin-bottom: 0.2rem !important;
        }
        h3 {
            margin-top: 0.2rem !important;
            margin-bottom: 0.3rem !important;
        }
        section[data-testid="stSidebar"] {
            width: 250px !important;
        }
    </style>
    """, unsafe_allow_html=True)

    st.sidebar.header("画面切替")
    view_mode = st.sidebar.radio("画面選択", ["標準出馬表", "分析出馬表"], index=0)

    st.title(f"🏇 出馬表 ({view_mode})")

    dates = fetch_dates()
    if not dates:
        st.error("開催日データが見つかりませんでした。")
        st.stop()

    st.sidebar.header("レース選択")
    selected_date = st.sidebar.selectbox(
        "開催日", 
        options=dates, 
        index=0,
        format_func=format_date_label
    )

    races_dict = get_races_for_date(selected_date)
    if not races_dict:
        st.sidebar.warning("該当日付のレースがありません。")
        st.stop()

    selected_race_id = st.sidebar.selectbox(
        "対象レース", 
        options=list(races_dict.keys()), 
        index=None,
        placeholder="対象レースを選択してください",
        format_func=lambda x: races_dict.get(x, x)
    )

    if not selected_race_id:
        st.info("👈 サイドバーから「対象レース」を選択してください。")
        st.stop()

    target_race_df = load_single_race_card(selected_race_id)

    if target_race_df.empty:
        st.error("レースデータの読み込みに失敗しました。")
        st.stop()

    target_cond = target_race_df['cond_code_youngest'].iloc[0] if 'cond_code_youngest' in target_race_df.columns else None
    target_grade = target_race_df['grade_code'].iloc[0] if 'grade_code' in target_race_df.columns else None
    target_course = target_race_df['course_code'].iloc[0] if 'course_code' in target_race_df.columns else None
    target_rank, target_class_name = get_class_info(target_cond, target_grade, target_course)

    if view_mode == "標準出馬表":
        render_standard_card(target_race_df, selected_date, target_rank, target_class_name)
    else:
        render_analysis_card(target_race_df, selected_date)

if __name__ == '__main__':
    main()

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
from fast_loader import get_available_dates, get_races_for_date, load_single_race_card
from past_races_loader import load_past_races_for_horses

WEEKDAY_JP = ['月', '火', '水', '木', '金', '土', '日']

def format_date_label(d_str):
    try:
        dt = datetime.strptime(d_str, '%Y-%m-%d')
        w = WEEKDAY_JP[dt.weekday()]
        return f"{d_str} ({w})"
    except Exception:
        return d_str

def calc_interval_info(recent_date_str, past_date_str):
    """
    recent_date_str: より新しい日付 (例: 当日または前走日)
    past_date_str: より古い過去走日付
    戻り値: (char, style_str)
    """
    if not recent_date_str or not past_date_str:
        return " ", "text-align: center;"
    try:
        dt_recent = datetime.strptime(str(recent_date_str).strip(), '%Y-%m-%d')
        dt_past = datetime.strptime(str(past_date_str).strip(), '%Y-%m-%d')
        days = (dt_recent - dt_past).days
        
        if days <= 9:
            # 9日以内: 黄色 (=)
            return "=", "background-color: #FFF59D; color: #000; font-weight: bold; text-align: center;"
        elif 70 <= days < 182:
            # 10週間(70日)〜半年(182日)未満: 黄緑 (+)
            return "+", "background-color: #AED581; color: #000; font-weight: bold; text-align: center;"
        elif days >= 182:
            # 半年以上: 赤 (+)
            return "+", "background-color: #EF5350; color: #FFF; font-weight: bold; text-align: center;"
        else:
            return " ", "text-align: center;"
    except Exception:
        return " ", "text-align: center;"

def get_track_dist_label(track_code, distance):
    """
    track_code: トラックコード (10~22: 芝, その他: ダート)
    distance: 距離 (例: 1600, 1150)
    戻り値: (label_text, style_str)
    """
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

# クラス情報およびランク判定
def get_class_info(cond_code, grade_code, course_code=None):
    cond = str(cond_code).strip() if cond_code is not None else ''
    grade = str(grade_code).strip() if grade_code is not None else ''
    c_code = str(course_code).strip() if course_code is not None else ''
    
    # 1. 地方競馬場（JRAコースコード 01~10 以外）の最優先判定
    is_local = False
    if c_code:
        try:
            c_num = int(c_code)
            if c_num not in range(1, 11):
                is_local = True
        except ValueError:
            pass

    if is_local:
        # 地方競馬の交流重賞 (A, B, C, D, G, H, L) ➔ OP (5.0)
        # ※ grade='E' (地方の一般特別) は OP にはせず 地方 (0.5) と判定
        if grade in ['A', 'B', 'C', 'D', 'G', 'H', 'L']:
            return 5.0, 'OP'
        # グレードコードのない地方競馬一般戦・一般特別 ➔ 地方 (0.5)
        return 0.5, '地方'

    # 2. JRAの特定条件クラス (1勝〜3勝・未勝利)
    if cond in ['701', '702', '703']:
        return 1.0, '未勝利/新馬'
    elif cond in ['003', '004', '005']:
        return 2.0, '1勝クラス'
    elif cond in ['007', '008', '009', '010']:
        return 3.0, '2勝クラス'
    elif cond in ['014', '015', '016']:
        return 4.0, '3勝クラス'

    # 3. JRAのオープン・重賞競走 (グレードあり または 条件000/999)
    if grade in ['A', 'B', 'C', 'D', 'E', 'G', 'H', 'L'] or cond in ['000', '999']:
        return 5.0, 'OP'

    return 3.0, '一般'

# 馬体重および増減フォーマット関数
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

# 単勝オッズフォーマット関数
def format_win_odds(odds_raw):
    o_str = str(odds_raw).strip() if pd.notna(odds_raw) else ''
    if not o_str.isdigit() or int(o_str) == 0:
        return ""
    val = float(int(o_str)) / 10.0
    return f"{val:.1f}"

# 斤量フォーマット関数 (型安全保護)
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

def main():
    st.set_page_config(page_title="出馬表", layout="wide")

    # 上部余白圧縮およびサイドバー拡張のカスタムCSS
    st.markdown("""
    <style>
        header[data-testid="stHeader"] {
            background-color: transparent !important;
        }
        .block-container {
            padding-top: 2.2rem !important;
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
            width: 380px !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # メインタイトル
    st.title("🏇 出馬表")

    dates = fetch_dates()
    if not dates:
        st.error("開催日データが見つかりませんでした。")
        st.stop()

    st.sidebar.header("レース選択")
    # 日付選択ドロップダウン (曜日のフォーマットを適用)
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
        format_func=lambda x: races_dict[x]
    )

    target_race_df = load_single_race_card(selected_race_id)

    if target_race_df.empty:
        st.error("レースデータの読み込みに失敗しました。")
        st.stop()

    # 当レースのクラス情報
    target_cond = target_race_df['cond_code_youngest'].iloc[0] if 'cond_code_youngest' in target_race_df.columns else None
    target_grade = target_race_df['grade_code'].iloc[0] if 'grade_code' in target_race_df.columns else None
    target_course = target_race_df['course_code'].iloc[0] if 'course_code' in target_race_df.columns else None
    target_rank, target_class_name = get_class_info(target_cond, target_grade, target_course)

    race_title = races_dict[selected_race_id]
    st.subheader(f"📊 {format_date_label(selected_date)} {race_title} 【クラス判定: {target_class_name}】")

    # 凡例表示
    st.markdown("""
    <div style="margin-bottom: 8px; padding: 8px 12px; background-color: #f8f9fa; border-radius: 8px; border: 1px solid #e9ecef;">
        <div style="font-weight: bold; font-size: 0.9em; color: #343a40;">
            💡 過去5走（前走〜５走前）セルの表記: <span style="color: #0d6efd; font-weight: bold;">[コース・距離][自馬の着順](前走1着馬の次走着順-出走馬の次走1着数-2,3着数)</span> （例: 芝16 1着(2-1-2) = 芝1600m走、自馬1着、前走1着馬の次走2着、出走馬のうち1頭が次走1着、2頭が次走2,3着）
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 過去5走データロード
    horse_id_list = target_race_df['horse_id'].unique().tolist()
    past_df = load_past_races_for_horses(horse_id_list, selected_date)

    # 馬ごとに過去5走の「自馬着順 ＋ 他馬の次走1着数-2,3着数」とクラス比較情報をマッピング
    past_5_map = {}
    if not past_df.empty:
        for h_id, group in past_df.groupby('horse_id'):
            past_5_map[h_id] = {}
            for _, row in group.iterrows():
                rn = int(row['rn'])
                if 1 <= rn <= 5:
                    # 自馬の着順 (0や空は空白)
                    order = row.get('final_order', '')
                    if pd.notna(order) and str(order).strip().isdigit() and int(order) > 0:
                        self_order_str = f"{int(order)}着"
                    else:
                        self_order_str = ""
                        
                    winner_next = row.get('winner_next_order_code', '*')
                    other_1st = row.get('other_next_1st_count', 0)
                    other_23rd = row.get('other_next_23rd_count', 0)
                    other_top3 = row.get('other_next_top3_count', 0)
                    
                    if self_order_str:
                        cell_text = f"{self_order_str}({winner_next}-{other_1st}-{other_23rd})"
                    else:
                        cell_text = f"({winner_next}-{other_1st}-{other_23rd})"
                    
                    p_cond = row.get('cond_code_youngest', None)
                    p_grade = row.get('grade_code', None)
                    p_course = row.get('course_code', None)
                    p_rank, p_cname = get_class_info(p_cond, p_grade, p_course)
                    
                    if p_rank == target_rank:
                        comp = 'same'
                    elif p_rank < target_rank:
                        comp = 'lower'
                    else:
                        comp = 'higher'

                    track_code = row.get('track_code', None)
                    distance = row.get('distance', None)
                    td_label, td_style = get_track_dist_label(track_code, distance)
                    
                    past_5_map[h_id][rn] = {
                        'text': cell_text,
                        'comp': comp,
                        'class_name': p_cname,
                        'race_date': row.get('race_date', ''),
                        'track_dist_label': td_label,
                        'track_dist_style': td_style,
                        'winner_next_order_code': winner_next,
                        'other_1st_count': other_1st,
                        'other_23rd_count': other_23rd,
                        'other_top3_count': other_top3,
                        'other_next_ran': row.get('other_next_ran', 0)
                    }

    # 表示用データフレーム構築
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
        
        # 馬齢: 0 の場合は空白文字に置換
        raw_age = row.get('horse_age', '')
        if pd.notna(raw_age) and str(raw_age).strip().isdigit() and int(raw_age) > 0:
            age_str = str(int(raw_age))
        else:
            age_str = ""
            
        sex_age = f"{sex}{age_str}"
        
        # 型安全な斤量フォーマット
        weight_carried = format_weight_carried(row.get('weight_carried', ''))
        jockey = row['jockey_name_short'] if pd.notna(row['jockey_name_short']) else ''
        
        # 単勝オッズフォーマット
        odds_fmt = format_win_odds(row.get('win_odds', ''))

        # 馬体重フォーマット
        h_weight = format_horse_weight(
            row.get('horse_weight', ''), 
            row.get('weight_change_sign', ''), 
            row.get('weight_change', '')
        )

        # 過去5走 (前走=1, 2走前=2, 3走前=3, 4走前=4, 5走前=5)
        p_data = past_5_map.get(h_id, {})
        r1 = p_data.get(1, {'text': '', 'comp': 'none', 'race_date': '', 'track_dist_label': '', 'track_dist_style': ''})
        r2 = p_data.get(2, {'text': '', 'comp': 'none', 'race_date': '', 'track_dist_label': '', 'track_dist_style': ''})
        r3 = p_data.get(3, {'text': '', 'comp': 'none', 'race_date': '', 'track_dist_label': '', 'track_dist_style': ''})
        r4 = p_data.get(4, {'text': '', 'comp': 'none', 'race_date': '', 'track_dist_label': '', 'track_dist_style': ''})
        r5 = p_data.get(5, {'text': '', 'comp': 'none', 'race_date': '', 'track_dist_label': '', 'track_dist_style': ''})

        # 出走間隔の判定 (0:当日-前走, 1:前走-2走前, 2:2走前-3走前, 3:3走前-4走前, 4:4走前-5走前)
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

    # HTMLテーブル描画（間隔セル幅を18pxに極小化、過去走セルを2分割）
    color_map = {
        'same': 'background-color: #FFF59D; color: #000; font-weight: bold;',
        'lower': 'background-color: #81D4FA; color: #000; font-weight: bold;',
        'higher': 'background-color: #CE93D8; color: #000; font-weight: bold;',
        'none': ''
    }

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
/* 間隔セルの文字幅ぴったり極小固定 */
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
        
        c1_style = color_map.get(comp[1], '')
        c2_style = color_map.get(comp[2], '')
        c3_style = color_map.get(comp[3], '')
        c4_style = color_map.get(comp[4], '')
        c5_style = color_map.get(comp[5], '')

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

if __name__ == '__main__':
    main()

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
    st.set_page_config(page_title="新・出馬表（過去着順・他馬次走3着以内頭数・クラス別色分け）", layout="wide")

    # サイドバーの横幅をゆったり拡張するカスタムCSS
    st.markdown("""
    <style>
        section[data-testid="stSidebar"] {
            width: 380px !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # メインタイトル
    st.title("🏇 JRA 新・出馬表（過去着順 ＋ 他走馬の次走3着以内頭数）")

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
    <div style="margin-bottom: 15px; padding: 12px 18px; background-color: #f8f9fa; border-radius: 8px; border: 1px solid #e9ecef;">
        <div style="font-weight: bold; font-size: 0.95em; color: #343a40; margin-bottom: 6px;">
            💡 過去5走（前走〜５走前）セルの表記: <span style="color: #0d6efd; font-weight: bold;">[自馬の着順](前走1着馬の次走着順-他同走馬の次走1着数-2,3着数)</span> （例: 1着(2-1-2) = 自馬1着、前走1着馬の次走2着、他1頭が次走1着、2頭が次走2,3着。※次走4着以下・データ無は *）
        </div>
        <div style="display: flex; gap: 15px; align-items: center;">
            <span style="font-weight: bold; font-size: 0.9em; color: #495057;">クラス判定背景色:</span>
            <span style="background-color: #FFF59D; color: #333; padding: 4px 12px; border-radius: 4px; font-weight: bold; font-size: 0.85em; border: 1px solid #FBC02D;">■ 同じクラスのレース</span>
            <span style="background-color: #81D4FA; color: #333; padding: 4px 12px; border-radius: 4px; font-weight: bold; font-size: 0.85em; border: 1px solid #0288D1;">■ 下のクラスのレース (格下)</span>
            <span style="background-color: #CE93D8; color: #333; padding: 4px 12px; border-radius: 4px; font-weight: bold; font-size: 0.85em; border: 1px solid #7B1FA2;">■ 上のクラスのレース (格上)</span>
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
                    
                    past_5_map[h_id][rn] = {
                        'text': cell_text,
                        'comp': comp,
                        'class_name': p_cname,
                        'winner_next_order_code': winner_next,
                        'other_1st_count': other_1st,
                        'other_23rd_count': other_23rd,
                        'other_top3_count': other_top3,
                        'other_next_ran': row.get('other_next_ran', 0)
                    }

    # 表示用データフレーム構築
    card_rows = []
    comp_matrix = []
    
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
        r1 = p_data.get(1, {'text': '', 'comp': 'none'})
        r2 = p_data.get(2, {'text': '', 'comp': 'none'})
        r3 = p_data.get(3, {'text': '', 'comp': 'none'})
        r4 = p_data.get(4, {'text': '', 'comp': 'none'})
        r5 = p_data.get(5, {'text': '', 'comp': 'none'})

        card_rows.append({
            '馬番': int(h_num),
            '枠': bracket,
            '馬名': h_name,
            '性齢': sex_age,
            '騎手': jockey,
            '斤量': weight_carried,
            'オッズ': odds_fmt,
            '馬体重': h_weight,
            '前走': r1['text'],
            '２走前': r2['text'],
            '３走前': r3['text'],
            '４走前': r4['text'],
            '５走前': r5['text']
        })
        
        comp_matrix.append({
            1: r1['comp'],
            2: r2['comp'],
            3: r3['comp'],
            4: r4['comp'],
            5: r5['comp']
        })

    disp_df = pd.DataFrame(card_rows)
    disp_df = disp_df.sort_values('馬番').reset_index(drop=True)

    # ユーザー指示のカラム並び順: 枠, 馬番, 馬名, 性齢, 騎手, 斤量, オッズ, 馬体重, 前走, ２走前...
    show_cols = ['枠', '馬番', '馬名', '性齢', '騎手', '斤量', 'オッズ', '馬体重', '前走', '２走前', '３走前', '４走前', '５走前']
    show_df = disp_df[show_cols]

    def style_past_races_clean(df):
        styles = pd.DataFrame('', index=df.index, columns=df.columns)
        color_map = {
            'same': 'background-color: #FFF59D; color: #000; font-weight: bold; text-align: center;',   # 黄色
            'lower': 'background-color: #81D4FA; color: #000; font-weight: bold; text-align: center;',  # 水色
            'higher': 'background-color: #CE93D8; color: #000; font-weight: bold; text-align: center;', # 赤紫
            'none': 'text-align: center;'
        }
        for i in range(len(df)):
            styles.loc[i, '前走'] = color_map.get(comp_matrix[i][1], 'text-align: center;')
            styles.loc[i, '２走前'] = color_map.get(comp_matrix[i][2], 'text-align: center;')
            styles.loc[i, '３走前'] = color_map.get(comp_matrix[i][3], 'text-align: center;')
            styles.loc[i, '４走前'] = color_map.get(comp_matrix[i][4], 'text-align: center;')
            styles.loc[i, '５走前'] = color_map.get(comp_matrix[i][5], 'text-align: center;')
        return styles

    styled_df = show_df.style.apply(style_past_races_clean, axis=None)

    calc_height = (len(disp_df) + 1) * 38 + 10
    st.dataframe(styled_df, width="stretch", height=calc_height, hide_index=True)

    # 過去走詳細展開
    if not past_df.empty:
        st.markdown("---")
        st.subheader("📋 過去5走 成績および他同走馬の次走成績詳細")
        
        selected_horse_num = st.selectbox(
            "詳細を確認したい馬番を選択してください",
            disp_df['馬番'].tolist(),
            format_func=lambda x: f"馬番 {x} : {disp_df[disp_df['馬番']==x]['馬名'].values[0]}"
        )
        
        selected_h_df = disp_df[disp_df['馬番'] == selected_horse_num]
        if not selected_h_df.empty:
            h_name = selected_h_df['馬名'].values[0]
            h_id_match = target_race_df[target_race_df['horse_number'].astype(int) == selected_horse_num]['horse_id'].values
            if len(h_id_match) > 0:
                target_h_id = h_id_match[0]
                h_past = past_df[past_df['horse_id'] == target_h_id].sort_values('rn')
                if not h_past.empty:
                    st.markdown(f"##### 🐴 馬番 {selected_horse_num} {h_name} の過去5走詳細成績")
                    detail_rows = []
                    for _, row in h_past.iterrows():
                        p_cond = row.get('cond_code_youngest', None)
                        p_grade = row.get('grade_code', None)
                        p_course = row.get('course_code', None)
                        _, p_cname = get_class_info(p_cond, p_grade, p_course)
                        
                        order_val = row.get('final_order', '')
                        if pd.notna(order_val) and str(order_val).strip().isdigit() and int(order_val) > 0:
                            order_disp = f"{int(order_val)}着"
                        else:
                            order_disp = ""
                            
                        winner_next = row.get('winner_next_order_code', '*')
                        other_1st = row.get('other_next_1st_count', 0)
                        other_23rd = row.get('other_next_23rd_count', 0)
                        other_ran = row.get('other_next_ran', 0)
                        next_top3_disp = f"1着馬次走:{winner_next} / 他同走 1着:{other_1st}頭, 2,3着:{other_23rd}頭 ({other_ran}頭中)"
                        
                        detail_rows.append({
                            '走次': f"{row['rn']}走前",
                            '日付': row.get('race_date', ''),
                            '競馬場': row.get('course_name', ''),
                            '種別': row.get('track_type', ''),
                            '距離': f"{row.get('distance', '')}m",
                            'クラス': p_cname,
                            '自馬着順': order_disp,
                            '他同走馬の次走成績': next_top3_disp,
                            'タイム': row.get('time_fmt', ''),
                            '上がり3F': row.get('last_3f_fmt', ''),
                            '騎手': row.get('jockey_name_short', '')
                        })
                    detail_df = pd.DataFrame(detail_rows)
                    st.dataframe(detail_df.set_index('走次'), width="stretch")
                else:
                    st.info("過去走データがありません。")

if __name__ == '__main__':
    main()

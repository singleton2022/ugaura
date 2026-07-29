import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import plotly.express as px
import os
from fast_loader import get_available_dates, get_races_for_date, load_single_race_card
from past_races_loader import load_past_races_for_horses

# ==========================================
# 1. ページ設定とモデルロード
# ==========================================
st.set_page_config(page_title="JRA AI 予測XAIダッシュボード", layout="wide")

@st.cache_data(ttl=3600)
def fetch_dates():
    return get_available_dates()

@st.cache_resource
def load_models():
    booster_turf = lgb.Booster(model_file='lgbm_model_turf.txt') if os.path.exists('lgbm_model_turf.txt') else None
    booster_dirt = lgb.Booster(model_file='lgbm_model_dirt.txt') if os.path.exists('lgbm_model_dirt.txt') else None
    return booster_turf, booster_dirt

def predict_with_booster(booster, df):
    feature_names = booster.feature_name()
    pandas_cat = booster.pandas_categorical
    cat_cols = set(feature_names[:len(pandas_cat)])
    
    df_feat = df.copy()
    
    for col in feature_names:
        if col not in df_feat.columns:
            df_feat[col] = 0

    cat_list = feature_names[:len(pandas_cat)]
    for col in feature_names:
        if col in cat_cols:
            idx = cat_list.index(col)
            categories = [str(c) for c in pandas_cat[idx]]
            df_feat[col] = pd.Categorical(df_feat[col].astype(str), categories=categories)
        else:
            df_feat[col] = pd.to_numeric(df_feat[col], errors='coerce').fillna(0.0)
            
    return booster.predict(df_feat[feature_names])

# ==========================================
# 2. メインUIとサイドバー（オンデマンド抽出）
# ==========================================
st.title("🏇 JRA LambdaRank 予測XAIダッシュボード")

booster_turf, booster_dirt = load_models()
dates = fetch_dates()

if not dates:
    st.error("開催日データが見つかりませんでした。")
    st.stop()

st.sidebar.header("レース選択")
selected_date = st.sidebar.selectbox("開催日", dates, index=0)

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

track_type = target_race_df['track_type'].iloc[0]

if track_type == '芝' and booster_turf is not None:
    target_race_df['predict_score'] = predict_with_booster(booster_turf, target_race_df)
elif track_type == 'ダート' and booster_dirt is not None:
    target_race_df['predict_score'] = predict_with_booster(booster_dirt, target_race_df)
else:
    target_race_df['predict_score'] = 0.0

target_race_df['予測順位'] = target_race_df['predict_score'].rank(ascending=False, method='min').astype(int)
target_race_df = target_race_df.sort_values('予測順位')

radar_cols = {
    'avg_ucv_score_5': '総合力 (UCV)',
    'avg_ucv_3f_5': '瞬発力 (UCV_3F)',
    'dash_score_median': '先行力 (Dash)',
    'target_elo_rating': '相手関係 (Elo)',
    'jockey_top3_rate_100': '騎手力 (Jockey)'
}

# ==========================================
# 3. 画面描画（全幅出馬表・スクロールなし）
# ==========================================
st.subheader(f"📊 {selected_date} {races_dict[selected_race_id]} の予測結果")

st.markdown("##### 出走馬リスト（予測順位順）")

disp_df = target_race_df[['予測順位', 'horse_number', 'horse_name', 'weight_carried', 'jockey_name_short', 'predict_score', 'is_win'] + list(radar_cols.keys())].copy()
disp_df.columns = ['予測順位', '馬番', '馬名', '斤量', '騎手', 'AIスコア', '結果(1着=1)'] + list(radar_cols.values())
disp_df['馬番'] = disp_df['馬番'].astype(str)
disp_df['AIスコア'] = disp_df['AIスコア'].round(3)

selected_horses = st.multiselect(
    "比較する馬番を選択すると下の過去5走チャートに反映されます", 
    disp_df['馬番'].tolist(),
    default=disp_df['馬番'].tolist()[:3]
)

# 18頭フルゲートでもスクロールなしで全頭表示されるようテーブル高さを自動計算
calc_height = (len(disp_df) + 1) * 38 + 10
st.dataframe(disp_df.set_index('予測順位'), use_container_width=True, height=calc_height)

# ==========================================
# 4. UCV 過去5走 出馬表エリア
# ==========================================
st.markdown("---")
st.subheader("📋 UCV 過去5走 出馬表・成績詳細")

horse_id_list = target_race_df['horse_id'].unique().tolist()
past_df = load_past_races_for_horses(horse_id_list, selected_date)

if past_df.empty:
    st.warning("対象馬の過去5走データが見つかりませんでした。")
else:
    horse_info_map = target_race_df[['horse_id', 'horse_number', 'predict_score', '予測順位']].set_index('horse_id').to_dict(orient='index')
    past_df['current_horse_number'] = past_df['horse_id'].map(lambda x: horse_info_map.get(x, {}).get('horse_number', '?'))
    past_df['predict_rank'] = past_df['horse_id'].map(lambda x: horse_info_map.get(x, {}).get('予測順位', 99))
    
    tab1, tab2, tab3 = st.tabs([
        "🏇 馬別 過去5走詳細カード", 
        "📈 UCVスコア推移グラフ", 
        "📺 全頭 UCV過去5走マトリックス"
    ])
    
    with tab1:
        st.markdown("##### 選択馬の過去5走 成績一覧")
        active_horses = selected_horses if selected_horses else target_race_df['horse_number'].astype(str).tolist()[:3]
        
        for h_num in active_horses:
            h_df = past_df[past_df['current_horse_number'].astype(str) == str(h_num)].sort_values('rn')
            if not h_df.empty:
                h_name = h_df['horse_name'].iloc[0]
                p_rank = h_df['predict_rank'].iloc[0]
                with st.expander(f"🐴 馬番 {h_num} : {h_name} (予測 {p_rank} 位)", expanded=True):
                    show_cols = ['past_label', 'race_date', 'course_name', 'track_type', 'distance', 'final_order', 'time_fmt', 'last_3f_fmt', 'ucv_score', 'ucv_3f_score', 'jockey_name_short']
                    renamed_cols = ['走次', 'レース日付', '場所', '種別', '距離', '着順', 'タイム', '上がり3F', 'UCVスコア', 'UCV 3F', '騎手']
                    view_h_df = h_df[show_cols].copy()
                    view_h_df.columns = renamed_cols
                    st.dataframe(view_h_df.set_index('走次'), use_container_width=True)

    with tab2:
        st.markdown("##### 過去5走 UCVスコアの変動比較")
        chart_horses = selected_horses if selected_horses else target_race_df['horse_number'].astype(str).tolist()
        chart_df = past_df[past_df['current_horse_number'].astype(str).isin(chart_horses)].copy()
        
        if not chart_df.empty:
            fig_ucv = px.line(
                chart_df, 
                x='past_label', 
                y='ucv_score', 
                color='horse_name', 
                markers=True,
                title="UCV 総合スコア推移 (直近5走)",
                labels={'past_label': '走次', 'ucv_score': 'UCV Score', 'horse_name': '馬名'}
            )
            fig_ucv.update_layout(xaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_ucv, use_container_width=True)
            
            fig_3f = px.line(
                chart_df, 
                x='past_label', 
                y='ucv_3f_score', 
                color='horse_name', 
                markers=True,
                title="UCV 3F (瞬発力)スコア推移 (直近5走)",
                labels={'past_label': '走次', 'ucv_3f_score': 'UCV 3F Score', 'horse_name': '馬名'}
            )
            fig_3f.update_layout(xaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_3f, use_container_width=True)

    with tab3:
        st.markdown("##### 全出走馬 過去5走 UCVスコア マトリックス")
        matrix_df = past_df.pivot(index=['current_horse_number', 'horse_name', 'predict_rank'], columns='past_label', values='ucv_score').reset_index()
        matrix_df = matrix_df.sort_values('predict_rank')
        renamed_mat = {'current_horse_number': '馬番', 'horse_name': '馬名', 'predict_rank': '予測順位'}
        matrix_df = matrix_df.rename(columns=renamed_mat)
        st.dataframe(matrix_df.set_index('予測順位'), use_container_width=True)

st.markdown("---")
st.caption("※ 過去5走のUCVスコアは、該当レース日付より過去に走破したJRA公式レース成績から算出された補正Zスコアです。")

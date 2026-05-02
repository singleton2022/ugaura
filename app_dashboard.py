import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import plotly.graph_objects as go
import os
from data_loader import load_feature_matrix

# ==========================================
# 1. ページ設定とキャッシュ関数
# ==========================================
st.set_page_config(page_title="JRA AI 予測XAIダッシュボード", layout="wide")

@st.cache_data(ttl=3600) # メモリ効率のためデータを1時間キャッシュ
def load_and_prep_data(db_path='C:/sqlite/jra_race.db'):
    df = load_feature_matrix(db_path)
    # 直近のレースのみ抽出（動作を軽くするため2025年以降に限定）
    df = df[df['race_date'] >= '2026-01-01'].copy()
    
    # --- 修正: race_idの末尾2文字からレース番号を復元 ---
    df['race_number'] = df['race_id'].astype(str).str[-2:].astype(int)
    
    # 競馬場名のマッピング
    course_map = {'01':'札幌', '02':'函館', '03':'福島', '04':'新潟', '05':'東京', '06':'中山', '07':'中京', '08':'京都', '09':'阪神', '10':'小倉'}
    df['course_name'] = df['course_code'].astype(str).str.zfill(2).map(course_map).fillna('その他')
    df['track_type'] = np.where(pd.to_numeric(df['track_code']).between(10, 22), '芝', 'ダート')
    
    # レース名表示用カラム
    df['race_display'] = df['race_date'] + " " + df['course_name'] + " " + df['race_number'].astype(str) + "R (" + df['track_type'] + df['distance'].astype(str) + "m)"
    return df

@st.cache_resource
def load_models():
    # 芝・ダートそれぞれのBoosterをロード
    booster_turf = lgb.Booster(model_file='lgbm_model_turf.txt') if os.path.exists('lgbm_model_turf.txt') else None
    booster_dirt = lgb.Booster(model_file='lgbm_model_dirt.txt') if os.path.exists('lgbm_model_dirt.txt') else None
    return booster_turf, booster_dirt

def prepare_features(df):
    cat_features = ['course_code', 'track_code', 'turf_condition_code', 'dirt_condition_code', 'bracket_number', 'horse_number', 'sex_code', 'distance']
    for col in cat_features:
        if df[col].dtype.name == 'category':
            if -1 not in df[col].cat.categories:
                df[col] = df[col].cat.add_categories([-1])
            df[col] = df[col].fillna(-1)
        else:
            df[col] = df[col].fillna(-1).astype('category')
    return df, cat_features

# ==========================================
# 2. メインUIとデータ処理
# ==========================================
st.title("🏇 JRA LambdaRank 予測解析ダッシュボード")

# データロード
with st.spinner('データベースから特徴量を読み込んでいます...'):
    raw_df = load_and_prep_data()
    booster_turf, booster_dirt = load_models()

# サイドバー: レース選択
st.sidebar.header("レース選択")
dates = sorted(raw_df['race_date'].unique(), reverse=True)
selected_date = st.sidebar.selectbox("開催日", dates)

# 選択された日付のレース一覧を取得
day_races = raw_df[raw_df['race_date'] == selected_date]
race_options = day_races[['race_id', 'race_display']].drop_duplicates().set_index('race_id')['race_display'].to_dict()
selected_race_id = st.sidebar.selectbox("対象レース", options=list(race_options.keys()), format_func=lambda x: race_options[x])

# 選択されたレースのデータを抽出
target_race_df = day_races[day_races['race_id'] == selected_race_id].copy()
track_type = target_race_df['track_type'].iloc[0]

# 推論の実行
num_features = [
    'pace_score_top3', 'prev_ucv_score', 'max_ucv_score_5', 'avg_ucv_score_5', 
    'avg_ucv_3f_5', 'max_ucv_3f_5', 'ucv_3f_gap_avg_5', 'ucv_3f_rank',
    'dash_score_median', 'leader_margin', 'target_elo_rating', 'horse_age', 'weight_carried',
    'ucv_score_rank', 'prev_ucv_score_rank', 'max_ucv_score_5_rank', 'elo_rating_rank', 'dash_score_rank',
    'ucv_score_z', 'elo_rating_z', 'weight_carried_diff', 'jockey_win_rate_100', 'jockey_top3_rate_100', 
    'straight_length', 'elo_diff_from_mean', 'ucv_elo_gap', 'interval_days',
    'horse_weight_num', 'weight_change_num', 'carried_weight_ratio', 'weight_change_per_day', 'is_fatigue_loss',
    'is_jockey_changed', 'is_sameday_jockey_change', 'bracket_win_rate_highres', 'bracket_top3_rate_highres',
    'slope_count', 'woodchip_count', 'slope_per_day', 'woodchip_per_day',
    'slope_total_time_4f_1', 'slope_total_time_4f_2',
    'slope_lap_time_2f_1f_1', 'slope_lap_time_2f_1f_2',
    'slope_lap_time_1f_0m_1', 'slope_lap_time_1f_0m_2',
    'woodchip_total_time_5f_1', 'woodchip_total_time_5f_2',
    'woodchip_total_time_4f_1', 'woodchip_total_time_4f_2',
    'woodchip_total_time_2f_1', 'woodchip_total_time_2f_2',
    'woodchip_lap_time_1f_0m_1', 'woodchip_lap_time_1f_0m_2'
]

target_race_df, cat_cols = prepare_features(target_race_df)
features = cat_cols + num_features

if track_type == '芝' and booster_turf is not None:
    target_race_df['predict_score'] = booster_turf.predict(target_race_df[features])
elif track_type == 'ダート' and booster_dirt is not None:
    target_race_df['predict_score'] = booster_dirt.predict(target_race_df[features])
else:
    st.error("対応する推論モデルが見つかりません。")
    st.stop()

# 予測順位と結果の結合
target_race_df['予測順位'] = target_race_df['predict_score'].rank(ascending=False, method='min').astype(int)
target_race_df = target_race_df.sort_values('予測順位')

# レーダーチャート用の偏差値化（レース内での相対評価 0-100）
radar_cols = {
    'avg_ucv_score_5': '総合力 (UCV)',
    'avg_ucv_3f_5': '瞬発力 (UCV_3F)',
    'dash_score_median': '先行力 (Dash)',
    'target_elo_rating': '相手関係 (Elo)',
    'jockey_top3_rate_100': '騎手力 (Jockey)'
}

for col, name in radar_cols.items():
    min_val = target_race_df[col].min()
    max_val = target_race_df[col].max()
    if max_val == min_val:
        target_race_df[f'{name}_scaled'] = 50.0
    else:
        target_race_df[f'{name}_scaled'] = ((target_race_df[col] - min_val) / (max_val - min_val)) * 100

# ==========================================
# 3. 画面描画
# ==========================================
st.subheader(f"📊 {race_options[selected_race_id]} の予測結果")

col1, col2 = st.columns([1, 1])

# --- 左カラム: データテーブル ---
with col1:
    st.markdown("##### 出走馬リスト（予測順位順）")
    # 表示用データフレーム
    disp_df = target_race_df[['予測順位', 'horse_number', 'predict_score', 'is_win'] + list(radar_cols.keys())].copy()
    disp_df.columns = ['予測順位', '馬番', 'AIスコア', '結果(1着=1)'] + list(radar_cols.values())
    disp_df['馬番'] = disp_df['馬番'].astype(str)
    disp_df['AIスコア'] = disp_df['AIスコア'].round(3)
    
    # 馬の選択（複数選択可）
    selected_horses = st.multiselect(
        "比較する馬番を選択してください（レーダーチャートに反映）", 
        disp_df['馬番'].tolist(),
        default=disp_df['馬番'].tolist()[:3] # デフォルトは上位3頭
    )
    
    st.dataframe(disp_df.set_index('予測順位'), use_container_width=True)

# --- 右カラム: レーダーチャート ---
with col2:
    st.markdown("##### 能力ベクトル比較 (レース内相対値 0-100)")
    if selected_horses:
        fig = go.Figure()
        categories = list(radar_cols.values())
        
        for horse in selected_horses:
            horse_data = target_race_df[target_race_df['horse_number'].astype(str) == horse].iloc[0]
            values = [horse_data[f'{cat}_scaled'] for cat in categories]
            # 閉じた多角形にするため先頭の値を末尾に追加
            values.append(values[0])
            cat_plot = categories + [categories[0]]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=cat_plot,
                fill='toself',
                name=f"馬番 {horse} (予測{horse_data['予測順位']}位)"
            ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("馬番を選択するとチャートが表示されます。")

st.markdown("---")
st.caption("※レーダーチャートの値は、該当レースに出走するメンバー内での偏差（Min-Max正規化）を表しています。100がメンバー中トップ、0がメンバー中最下位です。")
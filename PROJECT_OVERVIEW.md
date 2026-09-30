# Ugaura プロジェクト設計・実装仕様書（全体概要とソース状態記録）

更新日: 2026年9月28日
対象リポジトリ: singleton2022/ugaura (c:\Ugaura\Python)

---

# 1. プロジェクト概要

Ugaura（宇賀浦式 AI競馬予測システム）は、長年培われた競馬予想ノウハウ（減点方式・展開予想・トラックバイアス判定）と、統計的手法に基づく客観指数（UCV・UCV 3F・Eloレーティング）、および現代の機械学習アルゴリズム（LightGBM による順位学習 LambdaRank）を有機的に統合した競馬予測・投資支援プラットフォームです。

単なる勝敗予測にとどまらず、オッズと推定勝率から算出される「期待値（Expected Value: EV）」に基づいて購入推奨馬を抽出し、トータル回収率120%超の達成を目指す投資システムとして設計されています。

---

# 2. 基本設計思想と中核理論

### 2-1. 減点方式と展開予想のシステム化
競馬は「強い馬が勝つ」のではなく「弱点の少ない馬が勝つ」という基本哲学に基づきます。
ポジティブな加点要素ではなく、致命的な敗因となりうるリスク要因を減点方式で徹底排除します。

主な減点対象とAI特徴量へのマッピング:
- 前走逃げて好走した馬の反動減点: レース内の逃げ・先行馬構成（lead_horse_count_in_race）や真の逃げ馬フラグ（is_true_escape_prev）で展開崩れリスクを判定。
- トラック転向（初芝・初ダート）: 当該トラックのEloレーティング初期値（1000.0）による自動除外。
- 距離延長ローテーション: 前走距離との差分（distance_change）による折り合い・スタミナ消耗のペナルティ。
- 昇級初戦の壁: 前走クラスとの比較（is_promoted）による相手強化ペナルティ。
- 所属トレセン（西高東低）: 調教師の東西所属コード（affiliation_code）による水準差の評価。
- 長期休み明け・短期疲労: レース間隔日数（interval_days）、1日あたり体重増減率（weight_change_per_day）、短期疲労フラグ（is_fatigue_loss）。
- コース形状・枠順バイアス: 直線長（straight_length）、急坂適性（is_steep_slope, steep_slope_top3_rate）、枠順勝率（bracket_win_rate_highres）、小回り×先行力（interaction_small_turn_lead）。

### 2-2. 統計指数（絶対評価と相対評価の分離）
- UCV (Universal Course Value):
  コース・距離・クラスごとの良馬場基準タイム（base_time）と、当日の馬場差（track_bias: 20%トリム平均＋縮小推定）を用い、純粋な走破タイムから算出した能力偏差値（Zスコア）。ペース補正は一切行わず、タイムの客観性を担保。
- UCV 3F:
  上がり3ハロン専用のZスコア指数。直線のキレ・瞬発力を全体UCVと独立して評価。
- 対戦Eloレーティング:
  全レース結果から「1対N 個別撃破」Winner-vs-Losers モデルとクラス階層Kファクターを用いて算出された相対実力値。芝・ダート各5距離ノード（1200m, 1600m, 2000m, 2400m, 3000m）で管理され、ガウスカーネル補間によりレース距離に応じた target_elo_rating を導出。
- レースレベル (Pre-Score / Post-Score) および実走スコア:
  年齢区分（2歳・3歳・3歳以上）× クラス別の実測走破タイム差に基づく絶対基準点（class_base_score）をベースに、レース前メンバー水準（Pre-Score）と次3走実績（Post-Score）をベイズ平滑化により客観指数化。さらに上位5頭平均タイム差を加味した【実走スコア】を算出し、8段階ランク（S〜D）とヒートマップカラーで出馬表に可視化（詳細仕様: Plan/race_level_specification.md 参照）。

### 2-3. 二段階機械学習アーキテクチャ
- 第1段階（ペース予測回帰モデル）:
  レース全体の逃げ馬数、先行力上位3頭の平均ダッシュスコア、コース形状等から、レースの前後半ラップ差（前3F - 後3F: 前傾/後傾ラップ）を回帰予測（pred_lap_diff）。
- 第2段階（本番順位学習モデル）:
  pred_lap_diff をメタ特徴量として取り込み、芝とダートで物理的に分離した2つの LightGBM LambdaRank モデル（lgbm_model_turf.txt, lgbm_model_dirt.txt）でレース内の入着順位を学習。

### 2-4. 確率キャリブレーションと動的EV投資
LambdaRankの出力スコア（順位スコア）を、検証データ（2024年）を用いて確率へと変換。
- キャリブレーション手法:
  - Platt Scaling（ロジスティック回帰 ＋ Softmax）
  - Isotonic Regression（等張回帰）
  - Group-based Platt Scaling（良馬場 dry / 渋った馬場 wet 別ロジスティック回帰）
- コース別動的EVしきい値:
  回収率の出にくい難関コース（福島芝、小倉ダート等）には EV >= 1.30、中難度コースには EV >= 1.15、通常コースには EV >= 1.00 を要求して無駄な投資を抑制。

---

# 3. システムアーキテクチャとデータフロー

```mermaid
graph TD
    subgraph データインポート層
        JVLink[JRA-VAN / JV-Link] -->|32bit C++ sample1.exe| SQLite[(jra_race.db)]
        WebScrape[JRA公式馬場情報] -->|scrape_jra_cushion.py| CushionCSV[cushion/ CSV群]
    end

    subgraph 前処理・バッチ特徴量生成層
        SQLite -->|UgauraRatingSimulator.cpp| EloRating[feature_opponent_rating]
        SQLite -->|calc_ucv.py| UCVTables[horse_race_ucv / horse_race_ucv_3f]
        SQLite -->|generate_pace_features.py| PaceTables[feature_dash_score / feature_pace_prediction]
        SQLite -->|calc_race_level.py| RaceLevel[race_level テーブル]
        EloRating --> SQLite
        UCVTables --> SQLite
        PaceTables --> SQLite
        RaceLevel --> SQLite
    end

    subgraph 学習・検証パイプライン
        SQLite -->|data_loader.py + DuckDB Range Join| FeatureMatrix[統合特徴量マトリクス]
        FeatureMatrix -->|train_pace_model.py| PaceModel[lgbm_pace_model.txt]
        PaceModel -->|pred_lap_diff 推論| FeatureMatrix
        FeatureMatrix -->|train_model.py| TurfDirtModels[lgbm_model_turf.txt / lgbm_model_dirt.txt]
        TurfDirtModels -->|backtest_lambdarank.py| BacktestReport[バックテスト回収率評価]
    end

    subgraph リアルタイム運用・UI層
        JVLink -->|realtime_loader.py| SQLite
        SQLite -->|realtime_predict.py| LivePredict[リアルタイム推奨馬・期待値判定]
        SQLite -->|fast_loader.py| StreamlitApp[app_new_race_card.py / app_dashboard.py]
    end
```

---

# 4. 主要ソースファイル一覧と詳細仕様

### 4-1. データ準備・特徴量生成モジュール

- data_loader.py:
  システム全体の基幹データローダー。
  SQLiteからレース・馬情報、UCV、Elo、騎手勝率、ペース予測特徴量を一括取得。
  DuckDBのクエリエンジンを活用し、約500万件の坂路・ウッドチップ調教履歴から直近14日間の自己ベスト比率やラスト加速ラップ等の特徴量を高速に非等価結合（Range Join）。
  lgbm_pace_model.txt を用いて各レースの前後半ラップ差（pred_lap_diff）を推論し、フラットな学習用DataFrameを返却。

- calc_ucv.py:
  UCVおよびUCV 3Fの完全再計算スクリプト（週次実行）。
  2019年6月以降の良馬場走破タイムからコース・クラス別基準タイムを算出（トリム平均＋標準偏差）。
  同日・同トラックの偏差から20%トリム平均と縮小推定（alpha=3.0）で当日の馬場差を導出し、各馬の補正走破タイムからZスコアを算出してDBへ保存。

- generate_pace_features.py:
  各馬の初角ポジション指数および過去5戦のダッシュ力中央値を計算（feature_dash_score）。
  レース全体のダッシュ力上位3頭平均（pace_score_top3）および逃げ候補の単騎逃げ余裕度（leader_margin）を計算して feature_pace_prediction へ保存（週次実行）。

- calc_race_level.py:
  レースレベル（Pre-Score / Post-Score / Current-Score）の一括高速集計モジュール（DuckDB活用）。
  2015年以降の全確定走と未確定走を分離結合し、出走各馬の前走実績および次3走成績からレース前水準・事後確定水準を絶対値化（0〜100点）。結果を SQLite の race_level テーブルに格納。

- data_splitter.py:
  時系列分割ユーティリティ。
  Train（〜2023-12-31）、Validation（2024-01-01〜2024-12-31）、Test（2025-01-05〜）に厳密分割し未来データの混入を防止。

- fast_loader.py:
  Streamlitアプリ用の高速オンデマンドローダー。
  開催日一覧取得（get_available_dates）、指定日のレース一覧取得（get_races_for_date）、指定レースの出馬表単一ロード（load_single_race_card）を st.cache_data 付きで提供。

- past_races_loader.py:
  出馬表画面向けに、各出走馬の過去走（直近5走）の着順・タイム・UCV・上がり3Fなどを取得・集計。
  各過去走のレースレベル（race_level）と1〜5着馬平均走破タイムとの差分（上位5頭平均差）から【自馬パフォーマンス値（perf_score）】を算出し、8段階ランク（S, A+, A, B+, B, C+, C, D）および次3走成績（勝馬次走含む）を出馬表へ受け渡す。

- analysis_loader.py:
  分析出馬表画面向けに、各馬の距離適性、コース適性、急坂適性、季節実績、馬体重変動等のスタッツをオンデマンド集計。

### 4-2. 機械学習モジュール

- train_pace_model.py:
  レースの前後半ラップ差（front_3f - back_3f）を目的変数とするLightGBM回帰モデルを学習。
  コース形状、直線長、急坂、逃げ頭数、先行激化度などから展開バイアスを予測し、lgbm_pace_model.txt を出力。

- train_model.py:
  メイン予測モデルの学習スクリプト。
  data_loader から取得した特徴量マトリクスを芝・ダートに物理分割。
  LightGBM LambdaRank（目的変数: is_win, 順位評価指標: NDCG@1,3,5）で学習を行い、lgbm_model_turf.txt および lgbm_model_dirt.txt を出力。
  重要度（Gain）のTop15およびTest ROC-AUCを出力。

- backtest_lambdarank.py:
  テスト期間（2025年以降）に対する包括的バックテストエンジン。
  2024年の検証データを用いてキャリブレーション（Platt / Isotonic / Group-based Platt）を行い、推定勝率とオッズからEVを算出。
  コース別EVしきい値に基づき購入判定を行い、競馬場別・トラック別・距離別の的中率・回収率を集計出力。

### 4-3. 運用・リアルタイム予測モジュール

- realtime_loader.py:
  競馬開催当日に随時実行。
  32bit C++ネイティブアプリ（C:\Ugaura\Cpp\JravanImpor2025\Debug\sample1.exe /realtime YYYYMMDD）をキックし、JRA-VANからリアルタイムオッズ（O1）、馬体重（WH）、除外取消（AV）、騎手変更（JC）、天候馬場（WE）を取得してDBを更新。

- realtime_predict.py:
  最新のDB状態からリアルタイム予測・期待値再計算を実行。
  当日出走除外馬の勝率再分配処理、前半レース結果からのトラックバイアス自動検出、コース別EVしきい値判定を実施し、購入推奨馬を表示。
  コマンドラインオプションで競馬場（--course）やレース番号（--race_number）のピンポイント絞り込みが可能。

### 4-4. UI・可視化モジュール

- app_new_race_card.py:
  Streamlit製のWeb出馬表アプリ。
  「標準出馬表」（従来競馬新聞形式の着順・馬柱・オッズ表示）と「分析出馬表」（レーティング、UCV、脚質・調教・適性分析表示）の2画面を切り替え可能。

- app_dashboard.py:
  Streamlit製のXAIダッシュボード。
  予測スコア順位、レーダーチャート（総合力UCV、瞬発力UCV 3F、先行力Dash、相手関係Elo、騎手力Jockey）による特徴量可視化。

### 4-5. 馬場データ収集（cushion/）

- cushion/scrape_jra_cushion.py: JRA公式サイトから芝クッション値およびダート含水率をスクレイピング。
- cushion/generate_cushion_csv.py / generate_moisture_csv.py: 日付ごとの馬場データを年別CSV（クッション値YYYY.csv、ダート含水率YYYY.csv）に追記・整形。
- cushion/maintain_track_data.py: 過去データのフォーマット統一とメンテナンス。

---

# 5. データベース構成（SQLite: C:/Ugaura/sqlite/jra_race.db）

### 5-1. JRA-VAN 生データ系テーブル
- race_detail: レース条件、開催情報（年、月日、場コード、回、日、R、トラック、距離、馬場状態、天候、クラス、グレード、発走時刻など）。
- horse_race_info: 各馬の出走記録（血統登録番号、馬名、馬番、枠番、性齢、斤量、騎手コード、単勝オッズ、確定着順、走破タイム、上がり3Fタイム、通過順位など）。
- race_lap_time: レースのハロン別ラップタイム（1Fごとのタイム）。
- slope_training: 坂路調教記録（調教日付、血統番号、4Fタイム、2F-1Fラップ、1Fタイムなど）。
- woodchip_training: ウッドチップ調教記録（調教日付、血統番号、5F/4F/2Fタイム、1Fタイムなど）。

### 5-2. 統計指標・特徴量テーブル
- ucv_base_time / ucv_3f_base_time: コース・トラック・距離・クラス別の基準走破タイムおよび基準上がり3Fタイム。
- ucv_track_bias / ucv_3f_track_bias: 開催日・競馬場・トラックごとの算出馬場差。
- horse_race_ucv / horse_race_ucv_3f: 全出走記録に対する補正タイムおよび UCV / UCV 3F Zスコア。
- feature_opponent_rating: C++側で算出された各馬各レース時点の芝・ダート各距離Eloレーティング。
- feature_dash_score: 各馬の初角ポジション指数および過去5走ダッシュ力中央値。
- feature_pace_prediction: レース全体の先行激化度（上位3頭平均）および単騎逃げ余裕度。

---

# 6. 定常運用手順（マニュアル）

### 6-1. 毎週実施するバッチ処理（新規レース確定後）
1. C++ レーティングシミュレータの実行（全期間再計算）
   UgauraRatingSimulator.cpp の実行により feature_opponent_rating を再構築。
2. UCV指数の再計算
   python calc_ucv.py
3. ペース特徴量の再計算
   python generate_pace_features.py

### 6-2. モデル更新・再学習（随時・月次）
1. ペース予測回帰モデルの学習
   python train_pace_model.py
   -> lgbm_pace_model.txt が更新される。
2. 芝・ダート本番モデルの学習
   python train_model.py
   -> lgbm_model_turf.txt, lgbm_model_dirt.txt が更新される。
3. バックテストでの性能確認
   python backtest_lambdarank.py

### 6-3. 競馬開催当日の運用
1. 最新データの取得（オッズ・馬体重更新時）
   python realtime_loader.py --date YYYYMMDD
2. 推奨馬のリアルタイム予測
   - 全レース予測: python realtime_predict.py --date YYYYMMDD
   - 特定場のみ予測: python realtime_predict.py --date YYYYMMDD --course 東京
   - 特定レースのみ予測: python realtime_predict.py --date YYYYMMDD --course 東京 --race_number 11
   - 自動インポートを伴う予測: python realtime_predict.py --date YYYYMMDD --course 05 --import
3. 出馬表UIの閲覧
   streamlit run app_new_race_card.py

---

# 7. 現在の検証実績と主要課題

### 7-1. バックテスト実績（テスト期間: 2025-01-05 〜 2026-04-26）
- 総レース数: 4472レース
- 購入頭数: 1324頭（予測1位 かつ EV >= コース別しきい値）
- 的中頭数: 220頭（的中率: 16.62%）
- 全体回収率: 81.17%

高回収率条件:
- 新潟芝: 回収率 189.60% (直線が平坦で長く、差し有利バイアスが明瞭)
- 函館ダート: 回収率 181.10% (タフな路盤と調教負荷が直結)
- 新潟ダート: 回収率 150.90%
- 札幌芝: 回収率 140.30%
- 中京芝: 回収率 137.80%
- 中京ダート: 回収率 115.90%

低回収率条件（弱点コース）:
- 福島芝: 回収率 27.10% (小刻みなアップダウンと小回りで展開事故が多発)
- 小倉ダート: 回収率 31.40% (先行激化バイアス過多)
- 札幌ダート: 回収率 37.10%
- 中山芝: 回収率 43.00%

### 7-2. 現状の技術的課題
1. 予測実行時間のボトルネック:
   data_loader.py において、全レース（約50万件）と全調教履歴（500万件超）を毎回DuckDBで全結合しているため、数分を要する。差分ロードや中間特徴量キャッシュ（Parquet化）が必要。
2. サンプル数不足条件での過剰評価:
   小サンプル条件（長距離戦など）でたまたま的中したことによる「偽の高回収率」に惑わされない階層的キャリブレーション（Shrinkage）の導入。
3. 減点ロジックのさらなる統合:
   前走牝馬限定戦からのダート参戦、前走特殊馬場での好走（フロック勝ち割引）などの明示的特徴量化。

---

# 8. 今後の改善ロードマップ

1. 短期（実戦運用高速化・キャッシュ化）:
   - 特徴量マトリクスの中間Parquetキャッシュ保存機能の実装。
   - レース指定時の差分オンデマンド抽出処理の最適化。
   - 当日前半レースのトラックバイアス検知ロジックの高度化。

2. 中期（キャリブレーション高度化）:
   - 頭数別、クラス別のグループ化キャリブレーション検証。
   - Isotonic Regression による過大評価（偽のEV）抑制の本格実戦投入。
   - 地方競馬場・ローカルからの参戦ギャップ指数の追加。

3. 長期（モデル高度化・完全自動化）:
   - LambdaRank（順位学習）と勝率二値分類モデルのハイブリッド・アンサンブル。
   - コース形状の物理幾何パラメータ化（直線長・高低差・コーナー半径等の統一ベクトル化）。
   - 減点チェックリストの自動ポストプロセスフィルター統合。

# Ugaura プロジェクト AIアシスタント向けガイドライン (GEMINI.md)

このドキュメントは、AIアシスタント（Gemini / Antigravity）が本リポジトリで作業を開始する際に、プロジェクト構造と動作規約を迅速に理解するためのガイドです。

詳細なアーキテクチャや理論的背景は PROJECT_OVERVIEW.md を参照してください。

---

# 1. プロジェクトの概要

- プロジェクト名: Ugaura (宇賀浦式 AI競馬予測システム)
- 目的: 競馬データ分析ノウハウ（リスク要因の減点方式・展開予想）、客観統計指標（UCV・Eloレーティング）、および機械学習（LightGBM LambdaRank 順位学習）を統合し、回収率120%超を目指す競馬予測・投資システム。
- 実行環境: Windows, Python 3.10+, SQLite, DuckDB, LightGBM, Streamlit

---

# 2. 主要ファイルと役割

- PROJECT_OVERVIEW.md: システム全体の設計書、アーキテクチャ、全ソースの詳細仕様、バックテスト実績、改善ロードマップ。
- data_loader.py: 全レース・調教・統計指標を結合し、機械学習用の統合特徴量マトリクスを生成する中核モジュール（DuckDBによる高速Range Join）。
- calc_ucv.py: 走破タイムおよび上がり3Fの基準タイム・馬場差・UCV（能力偏差値Zスコア）の算出。
- calc_race_level.py: レースレベル（Pre-Score / Post-Score / Current-Score）の一括算出および race_level テーブル生成。
- generate_pace_features.py: 初角ポジション指数、ダッシュ力中央値、レース先行激化度の計算。
- train_pace_model.py: レース展開（前後半ラップ差 pred_lap_diff）を予測するLightGBM回帰モデルの学習。
- train_model.py: 芝・ダートを物理分離した本番用LightGBM LambdaRankモデル（lgbm_model_turf.txt, lgbm_model_dirt.txt）の学習。
- backtest_lambdarank.py: 2024年検証データを用いたキャリブレーション（Platt/Isotonic/Group-based）と、2025年以降のテストデータによる的中率・回収率バックテスト。
- realtime_loader.py: 32bit C++ sample1.exe を呼び出し、JRA-VANから当日の速報データ（オッズ、馬体重、除外、騎手変更、天候馬場）をSQLiteへ反映。
- realtime_predict.py: 最新DBから出走除外馬の勝率再分配、トラックバイアス検出、EV判定を行い、購入推奨馬を出力。
- app_new_race_card.py: Streamlit による Web出馬表（標準出馬表 / 分析出馬表）。
- fast_loader.py: Streamlit アプリ用のオンデマンド高速データローダー。
- past_races_loader.py: 出馬表向けの過去走（直近5走）取得および実走スコア（上位5頭平均差補正）算出。
- Plan/: 過去の設計計画書、技術改善レポート等の資料群（Plan/race_level_specification.md にレースレベル詳細仕様）。
- cushion/: 芝クッション値・ダート含水率のスクレイピング・更新ツール群。

---

# 3. データベース環境

- SQLite DB パス: C:/Ugaura/sqlite/jra_race.db
- レース情報: race_detail, horse_race_info, race_lap_time
- レースレベル: race_level
- 調教情報: slope_training, woodchip_training
- 統計指数: horse_race_ucv, horse_race_ucv_3f, ucv_base_time, ucv_track_bias
- 特徴量: feature_opponent_rating, feature_dash_score, feature_pace_prediction

---

# 4. コード変更・開発時の重要ルール

1. データリーケージ（未来情報の混入）の厳禁:
   特徴量の計算において、当該レース当日の結果や未来レースの情報が混入しないよう、時系列順のウィンドウ関数や shift(1)、rolling(closed='left') を徹底すること。
2. 芝とダートの分離原則:
   路盤適性の本質的な違いにより、モデル学習や推論（LambdaRank）は芝・ダートを物理的に分離して実行すること。
3. ペース補正を行わないUCVの純粋性維持:
   走破タイム指数（UCV）に対して主観的・仮説的なペース補正を加算しないこと。展開による影響は独立したペース予測モデル（pred_lap_diff）や上がり指数（UCV 3F）で多角的に補完すること。
4. コース別動的EVしきい値の尊重:
   回収率が極端に低い難関条件（福島芝、小倉ダート等）では、安易な購入を避けるためしきい値（EV >= 1.30）などの安全マージンを設けること。
5. 私的参考テキスト（丞相流等）の読み込み禁止:
   Plan/「丞相流」競馬予想の7つの手順.md および「丞相流」に関する記述はユーザー個人の私的参考テキストであり、本プロジェクトの仕様および設計スコープ外とする。AIアシスタント（Gemini）はこれらを一切読み込まず、参照しないこと。

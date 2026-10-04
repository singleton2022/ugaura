"""
calc_race_level.py: レースレベル（Pre-Score / Post-Score / Current-Score）算出モジュール

【概要】
1つのレースについて、全出走馬の前走成績（着順・着差・クラス）からレース前レベル（Pre-Score）を算出し、
レース後の毎週の次走成績（各馬の次走〜3走後までの着順・着差）から事後確定レースレベル（Post-Score）を集計・更新します。
DuckDBを用いて高速に一括算出し、SQLiteテーブル `race_level` に格納します。

【算出指標】
- pre_level_score (0〜100): レース前のメンバー水準（前走成績ベース）
- post_level_score (0〜100): レース後のメンバー実績（次3走実績ベース、ベイジアン平滑化）
- current_level_score (0〜100): 未出走走数に応じた現在進行形の総合スコア
- pre_level_rank / post_level_rank / current_level_rank: S(>=70), A(>=58), B(>=45), C(>=35), D(<35)
- next3_total_runs / next3_wins / next3_2nd / next3_3rd: 次3走の成績内訳
- winner_next_order: 当該レース勝ち馬の次走着順
"""

import sys
import io
import time
import argparse
import datetime
import sqlite3
import duckdb
import pandas as pd

# Windowsコンソール文字化け対策
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

DB_PATH = 'C:/Ugaura/sqlite/jra_race.db'

def create_target_table(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS race_level (
        race_id TEXT PRIMARY KEY,
        race_date TEXT,
        race_name TEXT,
        course_code TEXT,
        grade_code TEXT,
        class_rank REAL,
        starter_count INTEGER,
        pre_level_score REAL,
        pre_level_rank TEXT,
        post_level_score REAL,
        post_level_rank TEXT,
        current_level_score REAL,
        current_level_rank TEXT,
        next3_total_runs INTEGER,
        next3_wins INTEGER,
        next3_2nd INTEGER,
        next3_3rd INTEGER,
        winner_next_order INTEGER,
        updated_at TEXT
    );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_race_level_date ON race_level(race_date);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_race_level_current ON race_level(current_level_score);")
    conn.commit()
    conn.close()

def compute_race_levels(start_year=2015, db_path=DB_PATH):
    print(f"=== レースレベル算出開始 (対象: {start_year}年以降) ===")
    t0 = time.time()
    
    con = duckdb.connect()
    con.execute("INSTALL sqlite; LOAD sqlite;")
    con.execute(f"ATTACH '{db_path}' AS jra (TYPE SQLITE);")

    # クラス基準点（絶対下駄）SQL: 年齢区分（2歳・3歳・3歳以上）× クラス別の実測走破タイム差に基づく
    class_base_sql = """
    CASE 
        -- 2歳戦 (新馬・未勝利、1勝クラス、オープン・リステッド、G3・G2、G1)
        WHEN rd.race_type_code = '11' THEN
            CASE 
                WHEN rd.grade_code = 'A' THEN 63.0                            -- 2歳 G1 (阪神JF, 朝日杯FS, ホープフルS)
                WHEN rd.grade_code IN ('B', 'C') THEN 54.0                    -- 2歳 G3・G2
                WHEN rd.grade_code = 'L' OR rd.cond_code_youngest IN ('000', '999') THEN 49.0 -- 2歳 オープン・リステッド
                WHEN rd.cond_code_youngest IN ('003', '004', '005') THEN 44.0  -- 2歳 1勝クラス
                WHEN rd.cond_code_youngest IN ('701', '702', '703') THEN 34.0  -- 2歳 新馬・未勝利
                ELSE 34.0
            END

        -- 3歳戦 (新馬・未勝利、1勝クラス、2勝クラス、オープン・リステッド、G3・G2、G1)
        WHEN rd.race_type_code = '12' THEN
            CASE 
                WHEN rd.grade_code = 'A' THEN 72.0                            -- 3歳 G1 (クラシック等)
                WHEN rd.grade_code IN ('B', 'C') THEN 63.5                    -- 3歳 G3・G2
                WHEN rd.grade_code = 'L' OR rd.cond_code_youngest IN ('000', '999') THEN 58.0 -- 3歳 オープン・リステッド
                WHEN rd.cond_code_youngest IN ('007', '008', '009', '010') THEN 55.5 -- 3歳 2勝クラス
                WHEN rd.cond_code_youngest IN ('003', '004', '005') THEN 48.5  -- 3歳 1勝クラス
                WHEN rd.cond_code_youngest IN ('701', '702', '703') THEN 38.0  -- 3歳 新馬・未勝利
                ELSE 38.0
            END

        -- 3歳以上戦 (1勝クラス、2勝クラス、3勝クラス、オープン・リステッド、G3・G2、G1)
        ELSE
            CASE 
                WHEN rd.grade_code = 'A' THEN 77.0                            -- 3歳以上 G1
                WHEN rd.grade_code IN ('B', 'C') THEN 71.5                    -- 3歳以上 G3・G2
                WHEN rd.cond_code_youngest IN ('000', '999') OR rd.grade_code = 'L' THEN 67.0 -- 3歳以上 オープン・リステッド
                WHEN rd.cond_code_youngest IN ('014', '015', '016') THEN 62.5 -- 3歳以上 3勝クラス
                WHEN rd.cond_code_youngest IN ('007', '008', '009', '010') THEN 56.5 -- 3歳以上 2勝クラス
                WHEN rd.cond_code_youngest IN ('003', '004', '005') THEN 50.0 -- 3歳以上 1勝クラス
                WHEN rd.cond_code_youngest IN ('701', '702', '703') THEN 38.0 -- 障害未勝利など
                ELSE 50.0
            END
    END
    """

    # クラス判定用SQL（前走との格上挑戦・格下判定用）
    class_case_sql = """
    CASE 
        WHEN rd.race_type_code = '11' THEN
            CASE 
                WHEN rd.grade_code = 'A' THEN 4.0
                WHEN rd.grade_code IN ('B', 'C') THEN 3.5
                WHEN rd.grade_code = 'L' OR rd.cond_code_youngest IN ('000', '999') THEN 3.0
                WHEN rd.cond_code_youngest IN ('003', '004', '005') THEN 2.0
                ELSE 1.0
            END
        WHEN rd.race_type_code = '12' THEN
            CASE 
                WHEN rd.grade_code = 'A' THEN 5.0
                WHEN rd.grade_code IN ('B', 'C') THEN 4.5
                WHEN rd.grade_code = 'L' OR rd.cond_code_youngest IN ('000', '999') THEN 4.0
                WHEN rd.cond_code_youngest IN ('007', '008', '009', '010') THEN 3.0
                WHEN rd.cond_code_youngest IN ('003', '004', '005') THEN 2.0
                ELSE 1.0
            END
        ELSE
            CASE 
                WHEN rd.grade_code = 'A' THEN 6.0
                WHEN rd.grade_code IN ('B', 'C') THEN 5.5
                WHEN rd.cond_code_youngest IN ('000', '999') OR rd.grade_code = 'L' THEN 5.0
                WHEN rd.cond_code_youngest IN ('014', '015', '016') THEN 4.0
                WHEN rd.cond_code_youngest IN ('007', '008', '009', '010') THEN 3.0
                WHEN rd.cond_code_youngest IN ('003', '004', '005') THEN 2.0
                ELSE 1.0
            END
    END
    """

    # タイム差のパース（秒換算）
    time_diff_case_sql = """
    CASE 
        WHEN hri.time_diff IS NULL OR trim(hri.time_diff) = '' THEN NULL
        WHEN starts_with(trim(hri.time_diff), '+') THEN TRY_CAST(substr(trim(hri.time_diff), 2) AS DOUBLE) / 10.0
        WHEN starts_with(trim(hri.time_diff), '-') THEN -TRY_CAST(substr(trim(hri.time_diff), 2) AS DOUBLE) / 10.0
        WHEN TRY_CAST(trim(hri.time_diff) AS DOUBLE) IS NOT NULL THEN TRY_CAST(trim(hri.time_diff) AS DOUBLE) / 10.0
        ELSE NULL
    END
    """

    print("Step 1: 出走馬履歴テーブルの作成（確定走と未確定走の分離）...")
    # 前走・次走の参照のため、指定年の1年前から抽出
    filter_year = max(1989, start_year - 1)
    
    # 確定走（正常完走または降着でタイム・着順が確定しているレース）
    con.execute(f"""
    CREATE TEMP TABLE finished_runs AS
    SELECT 
        (rd.year || rd.month_day || rd.course_code || rd.times || rd.day || rd.race_number) AS race_id,
        rd.year,
        CAST(rd.year || '-' || substr(rd.month_day, 1, 2) || '-' || substr(rd.month_day, 3, 2) AS DATE) AS race_date,
        rd.course_code,
        CAST(rd.race_number AS INTEGER) AS race_number,
        rd.race_name_main,
        rd.grade_code,
        {class_base_sql} AS class_base_score,
        {class_case_sql} AS class_rank,
        hri.blood_reg_number AS horse_id,
        hri.horse_name,
        hri.horse_number,
        TRY_CAST(hri.final_order AS INTEGER) AS final_order,
        hri.abnormality_code,
        {time_diff_case_sql} AS time_diff_sec
    FROM jra.race_detail rd
    JOIN jra.horse_race_info hri 
        ON rd.year = hri.year 
       AND rd.month_day = hri.month_day 
       AND rd.course_code = hri.course_code 
       AND rd.times = hri.times 
       AND rd.day = hri.day 
       AND rd.race_number = hri.race_number
    WHERE CAST(rd.year AS INTEGER) >= {filter_year}
      AND rd.course_code IN ('01','02','03','04','05','06','07','08','09','10')
      AND hri.horse_number != '00'
      AND TRY_CAST(hri.final_order AS INTEGER) > 0
      AND hri.running_time IS NOT NULL 
      AND hri.running_time != '0000'
      AND hri.abnormality_code IN ('0', '7')
    QUALIFY ROW_NUMBER() OVER (PARTITION BY race_id, horse_id ORDER BY horse_number) = 1;
    """)

    # 未確定走（当日レースなど結果未確定のレース。Pre-Score算出用）
    con.execute(f"""
    CREATE TEMP TABLE upcoming_runs AS
    SELECT 
        (rd.year || rd.month_day || rd.course_code || rd.times || rd.day || rd.race_number) AS race_id,
        rd.year,
        CAST(rd.year || '-' || substr(rd.month_day, 1, 2) || '-' || substr(rd.month_day, 3, 2) AS DATE) AS race_date,
        rd.course_code,
        CAST(rd.race_number AS INTEGER) AS race_number,
        rd.race_name_main,
        rd.grade_code,
        {class_base_sql} AS class_base_score,
        {class_case_sql} AS class_rank,
        hri.blood_reg_number AS horse_id,
        hri.horse_name,
        hri.horse_number,
        0 AS final_order,
        hri.abnormality_code,
        NULL::DOUBLE AS time_diff_sec
    FROM jra.race_detail rd
    JOIN jra.horse_race_info hri 
        ON rd.year = hri.year 
       AND rd.month_day = hri.month_day 
       AND rd.course_code = hri.course_code 
       AND rd.times = hri.times 
       AND rd.day = hri.day 
       AND rd.race_number = hri.race_number
    WHERE CAST(rd.year AS INTEGER) >= {filter_year}
      AND rd.course_code IN ('01','02','03','04','05','06','07','08','09','10')
      AND hri.horse_number != '00'
      AND (TRY_CAST(hri.final_order AS INTEGER) IS NULL 
           OR TRY_CAST(hri.final_order AS INTEGER) = 0 
           OR hri.running_time IS NULL 
           OR hri.running_time = '0000')
      AND hri.abnormality_code = '0'
    QUALIFY ROW_NUMBER() OVER (PARTITION BY race_id, horse_id ORDER BY horse_number) = 1;
    """)

    print("Step 2: 確定走のWindow結合 & 未確定走のASOF結合...")
    # 確定走のみで次3走（LEAD）を計算することで、未確定レースが次走として混入することを完全に防ぐ
    con.execute("""
    CREATE TEMP TABLE finished_history AS
    SELECT 
        r.*,
        LAG(race_id, 1) OVER w AS prev_race_id,
        LAG(final_order, 1) OVER w AS prev_order,
        LAG(time_diff_sec, 1) OVER w AS prev_time_diff,
        LAG(class_rank, 1) OVER w AS prev_class,
        LAG(abnormality_code, 1) OVER w AS prev_abnormality,
        
        LEAD(race_id, 1) OVER w AS next1_race_id,
        LEAD(final_order, 1) OVER w AS next1_order,
        LEAD(time_diff_sec, 1) OVER w AS next1_time_diff,
        LEAD(class_rank, 1) OVER w AS next1_class,
        LEAD(grade_code, 1) OVER w AS next1_grade,

        LEAD(race_id, 2) OVER w AS next2_race_id,
        LEAD(final_order, 2) OVER w AS next2_order,
        LEAD(time_diff_sec, 2) OVER w AS next2_time_diff,
        LEAD(class_rank, 2) OVER w AS next2_class,
        LEAD(grade_code, 2) OVER w AS next2_grade,

        LEAD(race_id, 3) OVER w AS next3_race_id,
        LEAD(final_order, 3) OVER w AS next3_order,
        LEAD(time_diff_sec, 3) OVER w AS next3_time_diff,
        LEAD(class_rank, 3) OVER w AS next3_class,
        LEAD(grade_code, 3) OVER w AS next3_grade
    FROM finished_runs r
    WINDOW w AS (PARTITION BY horse_id ORDER BY race_date ASC, race_number ASC);
    """)

    # 未確定走は確定走から最新の前走情報のみを取得し、次走はすべてNULLとする
    con.execute("""
    CREATE TEMP TABLE upcoming_history AS
    SELECT 
        u.*,
        f.race_id AS prev_race_id,
        f.final_order AS prev_order,
        f.time_diff_sec AS prev_time_diff,
        f.class_rank AS prev_class,
        f.abnormality_code AS prev_abnormality,
        
        NULL::TEXT AS next1_race_id,
        NULL::INTEGER AS next1_order,
        NULL::DOUBLE AS next1_time_diff,
        NULL::DOUBLE AS next1_class,
        NULL::TEXT AS next1_grade,

        NULL::TEXT AS next2_race_id,
        NULL::INTEGER AS next2_order,
        NULL::DOUBLE AS next2_time_diff,
        NULL::DOUBLE AS next2_class,
        NULL::TEXT AS next2_grade,

        NULL::TEXT AS next3_race_id,
        NULL::INTEGER AS next3_order,
        NULL::DOUBLE AS next3_time_diff,
        NULL::DOUBLE AS next3_class,
        NULL::TEXT AS next3_grade
    FROM upcoming_runs u
    ASOF LEFT JOIN finished_runs f
      ON u.horse_id = f.horse_id AND u.race_date > f.race_date;
    """)

    con.execute("""
    CREATE TEMP TABLE horse_history AS
    SELECT * FROM finished_history
    UNION ALL
    SELECT * FROM upcoming_history;
    """)

    print("Step 3: 各馬の前走スコアおよび次3走ポイントの算出...")
    con.execute("""
    CREATE TEMP TABLE horse_scores AS
    SELECT 
        hh.*,
        
        -- 前走スコア（未出走/初出走馬は 2.5 点の中立値）
        CASE 
            WHEN prev_race_id IS NULL THEN 2.5
            ELSE GREATEST(0.0, LEAST(10.0,
                (CASE 
                    -- 同クラスまたは格上からの出走（現級実績・降級）
                    WHEN prev_class >= class_rank THEN
                        (CASE 
                            WHEN prev_order = 1 THEN 4.5
                            WHEN prev_order = 2 THEN 3.5
                            WHEN prev_order = 3 THEN 2.5
                            WHEN prev_order IN (4, 5) THEN 1.5
                            WHEN prev_order BETWEEN 6 AND 9 THEN 0.5
                            ELSE 0.0 
                         END)
                    -- 下位クラスからの昇級初戦・格上挑戦
                    ELSE
                        (CASE 
                            WHEN prev_order = 1 THEN 2.5  -- 昇級初戦の勝ち上がり馬は標準（2.5点）を基準
                            WHEN prev_order = 2 THEN 1.5  -- 格上挑戦2着
                            WHEN prev_order = 3 THEN 1.0  -- 格上挑戦3着
                            ELSE 0.2 
                         END)
                 END)
                + (CASE 
                    WHEN prev_time_diff IS NULL THEN 0.0
                    WHEN prev_time_diff <= 0.2 THEN 1.5
                    WHEN prev_time_diff <= 0.5 THEN 0.8
                    WHEN prev_time_diff <= 1.0 THEN 0.3
                    WHEN prev_time_diff >= 2.0 THEN -1.0
                    ELSE 0.0
                   END)
                + (CASE 
                    WHEN prev_class = class_rank AND prev_order IN (2, 3) THEN 0.5
                    WHEN prev_class > class_rank AND prev_order <= 5 THEN 1.0 -- 格下げ・相手緩和
                    ELSE 0.0
                   END)
            ))
        END AS horse_pre_score,

        -- 次走1
        CASE 
            WHEN next1_race_id IS NULL OR next1_order IS NULL OR next1_order = 0 THEN NULL
            ELSE LEAST(10.0,
                (CASE 
                    WHEN next1_order = 1 THEN 5.5
                    WHEN next1_order = 2 THEN 3.8
                    WHEN next1_order = 3 THEN 2.5
                    WHEN next1_order IN (4, 5) THEN 1.2
                    WHEN next1_order BETWEEN 6 AND 9 THEN 0.4
                    ELSE 0.0
                 END)
                + (CASE 
                    WHEN next1_time_diff IS NOT NULL AND next1_time_diff <= 0.2 THEN 1.5
                    WHEN next1_time_diff IS NOT NULL AND next1_time_diff <= 0.5 THEN 0.5
                    ELSE 0.0
                   END)
                -- 上位クラス好走加点
                + (CASE 
                    WHEN next1_class > class_rank AND next1_order = 1 THEN
                        (CASE WHEN next1_class - class_rank >= 1.5 OR next1_grade IN ('A', 'B', 'C') THEN 3.0 ELSE 1.5 END)
                    WHEN next1_class > class_rank AND next1_order IN (2, 3) THEN
                        (CASE WHEN next1_class - class_rank >= 1.5 OR next1_grade IN ('A', 'B', 'C') THEN 2.0 ELSE 0.8 END)
                    ELSE 0.0
                   END)
                + (CASE WHEN final_order = 1 AND next1_order = 1 THEN 2.0 ELSE 0.0 END)
            )
        END AS next1_pts,

        -- 次走2
        CASE 
            WHEN next2_race_id IS NULL OR next2_order IS NULL OR next2_order = 0 THEN NULL
            ELSE LEAST(10.0,
                (CASE 
                    WHEN next2_order = 1 THEN 5.5
                    WHEN next2_order = 2 THEN 3.8
                    WHEN next2_order = 3 THEN 2.5
                    WHEN next2_order IN (4, 5) THEN 1.2
                    WHEN next2_order BETWEEN 6 AND 9 THEN 0.4
                    ELSE 0.0
                 END)
                + (CASE 
                    WHEN next2_time_diff IS NOT NULL AND next2_time_diff <= 0.2 THEN 1.5
                    WHEN next2_time_diff IS NOT NULL AND next2_time_diff <= 0.5 THEN 0.5
                    ELSE 0.0
                   END)
                -- 上位クラス好走加点
                + (CASE 
                    WHEN next2_class > class_rank AND next2_order = 1 THEN
                        (CASE WHEN next2_class - class_rank >= 1.5 OR next2_grade IN ('A', 'B', 'C') THEN 3.0 ELSE 1.5 END)
                    WHEN next2_class > class_rank AND next2_order IN (2, 3) THEN
                        (CASE WHEN next2_class - class_rank >= 1.5 OR next2_grade IN ('A', 'B', 'C') THEN 2.0 ELSE 0.8 END)
                    ELSE 0.0
                   END)
            )
        END AS next2_pts,

        -- 次走3
        CASE 
            WHEN next3_race_id IS NULL OR next3_order IS NULL OR next3_order = 0 THEN NULL
            ELSE LEAST(10.0,
                (CASE 
                    WHEN next3_order = 1 THEN 5.5
                    WHEN next3_order = 2 THEN 3.8
                    WHEN next3_order = 3 THEN 2.5
                    WHEN next3_order IN (4, 5) THEN 1.2
                    WHEN next3_order BETWEEN 6 AND 9 THEN 0.4
                    ELSE 0.0
                 END)
                + (CASE 
                    WHEN next3_time_diff IS NOT NULL AND next3_time_diff <= 0.2 THEN 1.5
                    WHEN next3_time_diff IS NOT NULL AND next3_time_diff <= 0.5 THEN 0.5
                    ELSE 0.0
                   END)
                -- 上位クラス好走加点
                + (CASE 
                    WHEN next3_class > class_rank AND next3_order = 1 THEN
                        (CASE WHEN next3_class - class_rank >= 1.5 OR next3_grade IN ('A', 'B', 'C') THEN 3.0 ELSE 1.5 END)
                    WHEN next3_class > class_rank AND next3_order IN (2, 3) THEN
                        (CASE WHEN next3_class - class_rank >= 1.5 OR next3_grade IN ('A', 'B', 'C') THEN 2.0 ELSE 0.8 END)
                    ELSE 0.0
                   END)
            )
        END AS next3_pts

    FROM horse_history hh;
    """)

    print("Step 4: 各馬のベスト走抽出（次3走のうち最良成績1走のみ選定）...")
    con.execute("""
    CREATE TEMP TABLE horse_next_runs AS
    SELECT 
        race_id,
        year,
        horse_id,
        horse_name,
        horse_number,
        final_order,
        horse_pre_score,
        1 AS run_num,
        next1_order AS n_order,
        next1_time_diff AS n_time_diff,
        next1_pts AS pts
    FROM horse_scores
    WHERE next1_pts IS NOT NULL
    UNION ALL
    SELECT 
        race_id,
        year,
        horse_id,
        horse_name,
        horse_number,
        final_order,
        horse_pre_score,
        2 AS run_num,
        next2_order AS n_order,
        next2_time_diff AS n_time_diff,
        next2_pts AS pts
    FROM horse_scores
    WHERE next2_pts IS NOT NULL
    UNION ALL
    SELECT 
        race_id,
        year,
        horse_id,
        horse_name,
        horse_number,
        final_order,
        horse_pre_score,
        3 AS run_num,
        next3_order AS n_order,
        next3_time_diff AS n_time_diff,
        next3_pts AS pts
    FROM horse_scores
    WHERE next3_pts IS NOT NULL;
    """)

    con.execute("""
    CREATE TEMP TABLE horse_best_runs AS
    SELECT *
    FROM horse_next_runs
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY race_id, horse_id 
        ORDER BY pts DESC, COALESCE(n_time_diff, 99.0) ASC, run_num ASC
    ) = 1;
    """)

    print("Step 5: レース単位の集計とスコアリング...")
    con.execute(f"""
    CREATE TEMP TABLE race_level_summary AS
    SELECT 
        hs.race_id,
        MAX(CAST(hs.race_date AS TEXT)) AS race_date,
        MAX(hs.race_name_main) AS race_name,
        MAX(hs.course_code) AS course_code,
        MAX(hs.grade_code) AS grade_code,
        MAX(hs.class_base_score) AS class_base_score,
        MAX(hs.class_rank) AS class_rank,
        COUNT(DISTINCT hs.horse_id) AS starter_count,
        
        AVG(hs.horse_pre_score) AS avg_pre_score,
        
        -- 次3走のいずれかを消化したユニーク出走頭数
        COUNT(hbr.horse_id) AS next3_total_runs,
        
        -- ベスト走での1着（勝ち上がり頭数）
        SUM(CASE WHEN hbr.n_order = 1 THEN 1 ELSE 0 END) AS next3_wins,

        -- ベスト走での2着好走頭数
        SUM(CASE WHEN hbr.n_order = 2 THEN 1 ELSE 0 END) AS next3_2nd,

        -- ベスト走での3着好走頭数
        SUM(CASE WHEN hbr.n_order = 3 THEN 1 ELSE 0 END) AS next3_3rd,

        -- 各馬一律重み 1.0
        COUNT(hbr.horse_id)::DOUBLE AS weight_sum,

        -- ベスト走の獲得ポイント合計
        COALESCE(SUM(hbr.pts), 0.0) AS points_sum,

        -- 当該レース1着馬の次走着順
        MAX(CASE WHEN hs.final_order = 1 AND hs.next1_order > 0 THEN hs.next1_order ELSE NULL END) AS winner_next_order

    FROM horse_scores hs
    LEFT JOIN horse_best_runs hbr 
      ON hs.race_id = hbr.race_id AND hs.horse_id = hbr.horse_id
    WHERE CAST(hs.year AS INTEGER) >= {start_year}
    GROUP BY hs.race_id;
    """)

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    df_final = con.execute(f"""
    SELECT 
        race_id,
        race_date,
        race_name,
        course_code,
        grade_code,
        class_rank,
        starter_count,
        
        -- レース前レベル (絶対値化): クラス基準点 + (avg_pre_score - 2.5) * 8.0
        ROUND(
            LEAST(100.0, GREATEST(0.0,
                class_base_score + (avg_pre_score - 2.5) * 8.0
            ))
        , 1) AS pre_level_score,
        
        -- 事後レベル (絶対値化): クラス基準点 + (actual_avg_pts - 2.7) * 8.0 (prior_mean=2.7, prior_weight=6.0)
        ROUND(
            LEAST(100.0, GREATEST(0.0,
                class_base_score + (((6.0 * 2.7 + points_sum) / (6.0 + weight_sum)) - 2.7) * 8.0
            ))
        , 1) AS post_level_score,

        -- 現在総合レベル (出走頭数の約80%消化で満了移行)
        ROUND(
            (1.0 - LEAST(1.0, weight_sum / GREATEST(1.0, starter_count * 0.8))) * 
            LEAST(100.0, GREATEST(0.0, class_base_score + (avg_pre_score - 2.5) * 8.0))
            + LEAST(1.0, weight_sum / GREATEST(1.0, starter_count * 0.8)) * 
            LEAST(100.0, GREATEST(0.0, class_base_score + (((6.0 * 2.7 + points_sum) / (6.0 + weight_sum)) - 2.7) * 8.0))
        , 1) AS current_level_score,

        next3_total_runs,
        next3_wins,
        next3_2nd,
        next3_3rd,
        winner_next_order,
        '{now_str}' AS updated_at

    FROM race_level_summary
    ORDER BY race_date ASC, race_id ASC;
    """).fetchdf()

    def get_rank(score):
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

    df_final['pre_level_rank'] = df_final['pre_level_score'].map(get_rank)
    df_final['post_level_rank'] = df_final['post_level_score'].map(get_rank)
    df_final['current_level_rank'] = df_final['current_level_score'].map(get_rank)

    # カラム並び順の整理
    cols_order = [
        'race_id', 'race_date', 'race_name', 'course_code', 'grade_code', 'class_rank',
        'starter_count', 'pre_level_score', 'pre_level_rank',
        'post_level_score', 'post_level_rank',
        'current_level_score', 'current_level_rank',
        'next3_total_runs', 'next3_wins', 'next3_2nd', 'next3_3rd',
        'winner_next_order', 'updated_at'
    ]
    df_final = df_final[cols_order]

    print(f"Step 5: SQLiteへの保存... (全 {len(df_final)} 件)")
    sqlite_conn = sqlite3.connect(db_path)
    cur = sqlite_conn.cursor()
    
    # 既存のテーブル作成
    create_target_table(db_path)
    
    # INSERT OR REPLACE
    insert_sql = """
    INSERT OR REPLACE INTO race_level (
        race_id, race_date, race_name, course_code, grade_code, class_rank,
        starter_count, pre_level_score, pre_level_rank,
        post_level_score, post_level_rank,
        current_level_score, current_level_rank,
        next3_total_runs, next3_wins, next3_2nd, next3_3rd,
        winner_next_order, updated_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """
    
    records = df_final.to_records(index=False).tolist()
    cur.executemany(insert_sql, records)
    sqlite_conn.commit()
    sqlite_conn.close()
    con.close()

    elapsed = time.time() - t0
    print(f"=== 算出完了! 処理レース数: {len(df_final)} 件 (所要時間: {elapsed:.2f}秒) ===")
    return df_final

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="レースレベル算出バッチ")
    parser.add_argument('--start-year', type=int, default=2020, help="算出開始年 (デフォルト: 2020)")
    args = parser.parse_args()

    compute_race_levels(start_year=args.start_year)

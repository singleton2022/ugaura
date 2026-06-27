import win32com.client
import sqlite3

def extract_field(byte_line, start_pos, length):
    """
    固定長バイトレコードから指定位置・指定長さのフィールドを抽出してShift-JISデコードする
    start_pos: 1-indexed (JRA-VAN仕様書のバイト位置に対応)
    """
    val = byte_line[start_pos - 1 : start_pos - 1 + length]
    return val.decode('cp932', errors='replace').strip()

def process_record(byte_line, cursor):
    """
    JRA-VAN速報レコードをパースし、SQLiteデータベースを更新する
    """
    if len(byte_line) < 10:
        return
        
    rec_id = extract_field(byte_line, 1, 2)
    
    # 共通開催キー
    year = extract_field(byte_line, 12, 4)
    month_day = extract_field(byte_line, 16, 4)
    course_code = extract_field(byte_line, 20, 2)
    times = extract_field(byte_line, 22, 2)
    day = extract_field(byte_line, 24, 2)
    
    if not (year.isdigit() and month_day.isdigit() and course_code.isdigit()):
        return

    if rec_id == "O1":
        # 単複オッズ
        race_number = extract_field(byte_line, 26, 2)
        for i in range(28):
            offset = 44 + i * 8
            horse_num = extract_field(byte_line, offset, 2)
            if not horse_num or horse_num == "00" or horse_num == "0":
                continue
            odds_str = extract_field(byte_line, offset + 2, 4)
            pop_str = extract_field(byte_line, offset + 6, 2)
            
            cursor.execute(
                """
                UPDATE horse_race_info 
                SET win_odds = ?, win_popularity = ? 
                WHERE year = ? AND month_day = ? AND course_code = ? AND times = ? AND day = ? AND race_number = ? AND horse_number = ?
                """,
                (odds_str, pop_str, year, month_day, course_code, times, day, race_number, horse_num.zfill(2))
            )
            
    elif rec_id == "WH":
        # 馬体重
        race_number = extract_field(byte_line, 26, 2)
        for i in range(18):
            offset = 36 + i * 45
            horse_num = extract_field(byte_line, offset, 2)
            if not horse_num or horse_num == "00" or horse_num == "0":
                continue
            weight = extract_field(byte_line, offset + 38, 3)
            sign = extract_field(byte_line, offset + 41, 1)
            change = extract_field(byte_line, offset + 42, 3)
            
            cursor.execute(
                """
                UPDATE horse_race_info 
                SET horse_weight = ?, weight_change_sign = ?, weight_change = ? 
                WHERE year = ? AND month_day = ? AND course_code = ? AND times = ? AND day = ? AND race_number = ? AND horse_number = ?
                """,
                (weight, sign, change, year, month_day, course_code, times, day, race_number, horse_num.zfill(2))
            )
            
    elif rec_id == "AV":
        # 出走取消・競走除外
        data_cat = extract_field(byte_line, 3, 1)
        race_number = extract_field(byte_line, 26, 2)
        horse_num = extract_field(byte_line, 36, 2)
        
        abn_code = "3" if data_cat == "1" else "4"
        cursor.execute(
            """
            UPDATE horse_race_info SET abnormality_code = ? 
            WHERE year = ? AND month_day = ? AND course_code = ? AND times = ? AND day = ? AND race_number = ? AND horse_number = ?
            """,
            (abn_code, year, month_day, course_code, times, day, race_number, horse_num.zfill(2))
        )
        
    elif rec_id == "JC":
        # 騎手変更
        race_number = extract_field(byte_line, 26, 2)
        horse_num = extract_field(byte_line, 36, 2)
        weight = extract_field(byte_line, 74, 3)
        jockey_code = extract_field(byte_line, 77, 5)
        jockey_name = extract_field(byte_line, 82, 34)
        
        jockey_name_clean = jockey_name.strip()
        
        cursor.execute(
            """
            UPDATE horse_race_info SET weight_carried = ?, jockey_code = ?, jockey_name_short = ? 
            WHERE year = ? AND month_day = ? AND course_code = ? AND times = ? AND day = ? AND race_number = ? AND horse_number = ?
            """,
            (weight, jockey_code, jockey_name_clean, year, month_day, course_code, times, day, race_number, horse_num.zfill(2))
        )
        
    elif rec_id == "WE":
        # 天候馬場状態
        weather = extract_field(byte_line, 35, 1)
        turf_cond = extract_field(byte_line, 36, 1)
        dirt_cond = extract_field(byte_line, 37, 1)
        
        cursor.execute(
            """
            UPDATE race_detail 
            SET weather_code = ?, turf_condition_code = ?, dirt_condition_code = ? 
            WHERE year = ? AND month_day = ? AND course_code = ? AND times = ? AND day = ?
            """,
            (weather, turf_cond, dirt_cond, year, month_day, course_code, times, day)
        )

def download_and_update_realtime(db_path, date_str):
    """
    JV-Link COMコントロールを使って指定日の速報データを取得し、SQLite DBを更新する
    オッズ取得を含め、JV-Linkからのロードで失敗した場合は処理を即時中断してエラーとする
    """
    print(f"[JV-Link] 接続処理を開始します (日付: {date_str})")
    
    try:
        jv_link = win32com.client.Dispatch("JVDTLab.JVLink.1")
    except Exception as e:
        print(f"[エラー] JVDTLab.JVLink.1 COMオブジェクトの生成に失敗しました: {e}")
        print("JV-Linkが正しくインストールされているか確認してください。")
        return False
        
    rc = jv_link.JVInit("UNKNOWN")
    if rc != 0:
        print(f"[エラー] JVInitに失敗しました (エラーコード: {rc})")
        return False
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 取得対象のdataspecリスト（変更情報、馬体重、単複枠オッズ）
    # ※本実装では計画に基づき、JV-Link公式からのみ全速報データを取得します。
    specs = [
        ("0B12", "当日変更情報(除外・騎手等)"),
        ("0B11", "当日の馬体重情報"),
        ("0B31", "当日の単複枠オッズ情報")
    ]
    
    try:
        cursor.execute("BEGIN TRANSACTION;")
        
        for dataspec, desc in specs:
            rc = jv_link.JVRTOpen(dataspec, date_str)
            if rc == 0:
                print(f"[JV-Link] {dataspec} ({desc}) 読み込み開始...")
                filename = ""
                count = 0
                while True:
                    read_buff = " " * 100000
                    read_size = 100000
                    ret, buff, size, filename = jv_link.JVRead(read_buff, read_size, filename)
                    if ret == 0:
                        break
                    if ret < 0:
                        print(f"[エラー] JVRead {dataspec} 読み込み中にエラーが発生しました (エラーコード: {ret})")
                        raise RuntimeError(f"JVRead error: {ret}")
                    
                    byte_line = buff[:ret].encode('cp932', errors='replace')
                    process_record(byte_line, cursor)
                    count += 1
                print(f"[JV-Link] {dataspec} 処理完了 ({count} レコード)")
                jv_link.JVClose()
            else:
                print(f"[エラー] JVRTOpen {dataspec} ({desc}) のオープンに失敗しました (エラーコード: {rc})")
                print("JRA-VANから必要なデータが取得できないため、リアルタイム更新処理を即時中断します。")
                raise RuntimeError(f"JVRTOpen error: {rc}")
            
        conn.commit()
        print("[JV-Link] すべてのリアルタイムデータ更新が正常に完了しました。")
        return True
    except Exception as e:
        conn.rollback()
        print(f"[エラー] リアルタイムデータ更新処理をロールバックして中止しました: {e}")
        return False
    finally:
        conn.close()

import sqlite3
import csv
import os
from datetime import date

# 競馬場コードと名称のマッピング
COURSE_NAME_MAP = {
    '01': '札幌', '02': '函館', '03': '福島', '04': '新潟', '05': '東京',
    '06': '中山', '07': '中京', '08': '京都', '09': '阪神', '10': '小倉'
}

def generate_moisture_csv():
    db_path = 'C:/Ugaura/sqlite/jra_race.db'
    
    # システム日付からデフォルト値(YYYYMMDD)を生成
    today_str = date.today().strftime('%Y%m%d')
    
    # 入力プロンプトから例を削除
    prompt_msg = f"作成する日付を入力してください [デフォルト: {today_str}]: "
    target_date = input(prompt_msg).strip()
    
    # 未入力の場合はデフォルト値を使用
    if not target_date:
        target_date = today_str
    elif len(target_date) != 8 or not target_date.isdigit():
        print("エラー: 日付は半角数字8桁で入力してください。")
        return

    year_val = target_date[:4]
    month_day_val = target_date[4:]

    if not os.path.exists(db_path):
        print(f"エラー: データベースファイル '{db_path}' が見つかりません。")
        return

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        query = """
            SELECT 
                h.year, h.month_day, h.course_code, h.times, h.day, h.race_number, h.horse_number
            FROM horse_race_info h
            INNER JOIN race_detail r 
                ON  h.year = r.year 
                AND h.month_day = r.month_day 
                AND h.course_code = r.course_code 
                AND h.times = r.times 
                AND h.day = r.day 
                AND h.race_number = r.race_number
            WHERE h.year = ? 
              AND h.month_day = ?
              AND CAST(r.track_code AS INTEGER) BETWEEN 23 AND 29
        """
        cursor.execute(query, (year_val, month_day_val))
        rows = cursor.fetchall()

        if not rows:
            print(f"{target_date} のダートレースのレコードは見つかりませんでした。")
            return

        course_codes = sorted(list(set(row[2] for row in rows)))
        
        moisture_map = {}
        print(f"\n--- {target_date} のダート含水率を入力してください ---")
        for code in course_codes:
            code_str = str(code).zfill(2)
            course_name = COURSE_NAME_MAP.get(code_str, "不明")
            
            val = input(f"競馬場コード {code_str} ({course_name}) のダート含水率: ").strip()
            moisture_map[code] = val

        output_file = f"ダート含水率{target_date}.csv"
        
        with open(output_file, mode='w', encoding='cp932', newline='') as f:
            writer = csv.writer(f)
            for row in rows:
                y, md, cc, t, d, rn, hn = row
                race_code_16 = (str(y) + str(md) + str(cc).zfill(2) + 
                                str(t).zfill(2) + str(d).zfill(2) + str(rn).zfill(2))
                horse_id_18 = race_code_16 + str(hn).zfill(2)
                
                moisture_val = moisture_map.get(cc, "")
                writer.writerow([horse_id_18, moisture_val])

        print(f"\nCSVファイルを作成しました: {output_file}")
        print(f"総行数: {len(rows)} (ダートレースのみ)")

    except sqlite3.Error as e:
        print(f"データベースエラー: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == '__main__':
    generate_moisture_csv()
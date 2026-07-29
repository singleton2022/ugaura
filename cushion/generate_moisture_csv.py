import sqlite3
import csv
import os
from datetime import date

COURSE_NAME_MAP = {
    '01': '\u672d\u5e4c', '02': '\u760d\u9928', '03': '\u798f\u5cf6', '04': '\u65b0\u6f5f', '05': '\u6771\u4eac',
    '06': '\u4e2d\u5c71', '07': '\u4e2d\u4eac', '08': '\u4eac\u90fd', '09': '\u962a\u795e', '10': '\u5c0f\u5009'
}

def get_dirt_courses(db_path, year_val, month_day_val):
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    query = '\n        SELECT DISTINCT h.course_code\n        FROM horse_race_info h\n        INNER JOIN race_detail r \n            ON  h.year = r.year \n            AND h.month_day = r.month_day \n            AND h.course_code = r.course_code \n            AND h.times = r.times \n            AND h.day = r.day \n            AND h.race_number = r.race_number\n        WHERE h.year = ? \n          AND h.month_day = ?\n          AND CAST(r.track_code AS INTEGER) BETWEEN 23 AND 29\n        ORDER BY h.course_code\n    '
    cursor.execute(query, (year_val, month_day_val))
    rows = cursor.fetchall()
    conn.close()
    return [row[0] for row in rows]

def get_dirt_race_rows(db_path, year_val, month_day_val):
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    query = '\n        SELECT \n            h.year, h.month_day, h.course_code, h.times, h.day, h.race_number, h.horse_number\n        FROM horse_race_info h\n        INNER JOIN race_detail r \n            ON  h.year = r.year \n            AND h.month_day = r.month_day \n            AND h.course_code = r.course_code \n            AND h.times = r.times \n            AND h.day = r.day \n            AND h.race_number = r.race_number\n        WHERE h.year = ? \n          AND h.month_day = ?\n          AND CAST(r.track_code AS INTEGER) BETWEEN 23 AND 29\n    '
    cursor.execute(query, (year_val, month_day_val))
    rows = cursor.fetchall()
    conn.close()
    return rows

def append_to_yearly_csv(yearly_csv_path, new_rows):
    existing_map = {}
    ordered_ids = []
    if os.path.exists(yearly_csv_path):
        with open(yearly_csv_path, mode='r', encoding='cp932', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or len(row) < 2:
                    continue
                hid, val = row[0], row[1]
                if hid not in existing_map:
                    ordered_ids.append(hid)
                existing_map[hid] = val

    for hid, val in new_rows:
        if hid not in existing_map:
            ordered_ids.append(hid)
        existing_map[hid] = val

    parent_dir = os.path.dirname(yearly_csv_path)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
        
    with open(yearly_csv_path, mode='w', encoding='cp932', newline='') as f:
        writer = csv.writer(f)
        for hid in ordered_ids:
            writer.writerow([hid, existing_map[hid]])

def generate_moisture_csv(target_date=None, moisture_map=None, drive_dir=None):
    if drive_dir is None:
        drive_dir = os.path.join('G:\\', '\u30de\u30a4\u30c9\u30e9\u30a4\u30d6', '\u30af\u30c3\u30b7\u30e7\u30f3\u5024')
    db_path = 'C:/Ugaura/sqlite/jra_race.db'
    today_str = date.today().strftime('%Y%m%d')

    if not target_date:
        prompt_msg = f"\u4f5c\u6210\u3059\u308b\u65e5\u4ed8\u3092\u5165\u529b\u3057\u3066\u304f\u3060\u3055\u3044 [\u30c7\u30d5\u30aa\u30eb\u30c8: {today_str}]: "
        target_date = input(prompt_msg).strip()
        if not target_date:
            target_date = today_str

    if len(target_date) != 8 or not target_date.isdigit():
        print("\u30a8\u30e9\u30fc: \u65e5\u4ed8\u306f\u534a\u89d2\u6570\u5b578\u6841\u3067\u5165\u529b\u3057\u3066\u304f\u3060\u3055\u3044\u3002")
        return False

    year_val = target_date[:4]
    month_day_val = target_date[4:]

    rows = get_dirt_race_rows(db_path, year_val, month_day_val)
    if not rows:
        print(f"{target_date} \u306e\u30c0\u30fc\u30c8\u30ec\u30fc\u30b9\u306e\u30ec\u30b3\u30fc\u30c9\u306f\u898b\u3064\u304b\u308a\u307e\u305b\u3093\u3067\u3057\u305f\u3002")
        return False

    if moisture_map is None:
        course_codes = sorted(list(set(row[2] for row in rows)))
        moisture_map = {}
        print(f"\n--- {target_date} \u306e\u30c0\u30fc\u30c8\u542b\u6c34\u7387\u3092\u5165\u529b\u3057\u3066\u304f\u3060\u3055\u3044 ---")
        for code in course_codes:
            code_str = str(code).zfill(2)
            course_name = COURSE_NAME_MAP.get(code_str, "\u4e0d\u660e")
            val = input(f"\u7af6\u99ac\u5834\u30b3\u30fc\u30c9 {code_str} ({course_name}) \u306e\u30c0\u30fc\u30c8\u542b\u6c34\u7387: ").strip()
            moisture_map[code] = val

    output_file = f"\u30c0\u30fc\u30c8\u542b\u6c34\u7387{target_date}.csv"
    output_path = os.path.join(os.path.dirname(__file__), output_file)

    generated_rows = []
    with open(output_path, mode='w', encoding='cp932', newline='') as f:
        writer = csv.writer(f)
        for row in rows:
            y, md, cc, t, d, rn, hn = row
            race_code_16 = (str(y) + str(md) + str(cc).zfill(2) + 
                            str(t).zfill(2) + str(d).zfill(2) + str(rn).zfill(2))
            horse_id_18 = race_code_16 + str(hn).zfill(2)
            moisture_val = moisture_map.get(cc, "")
            writer.writerow([horse_id_18, moisture_val])
            generated_rows.append((horse_id_18, moisture_val))

    print(f"\nCSV\u30d5\u30a1\u30a4\u30eb\u3092\u4f5c\u6210\u3057\u307e\u3057\u305f: {output_path}")
    print(f"\u7dcf\u884c\u6570: {len(rows)} (\u30c0\u30fc\u30c8\u30ec\u30fc\u30b9\u306e\u307f)")

    local_yearly = os.path.join(os.path.dirname(__file__), f"\u30c0\u30fc\u30c8\u542b\u6c34\u7387{year_val}.csv")
    append_to_yearly_csv(local_yearly, generated_rows)
    print(f"\u30ed\u30fc\u30ab\u30eb\u5e74\u9593\u30d5\u30a1\u30a4\u30eb\u306b\u53cd\u6620\u3057\u307e\u3057\u305f: {local_yearly}")

    if drive_dir and os.path.exists(drive_dir):
        drive_yearly = os.path.join(drive_dir, f"\u30c0\u30fc\u30c8\u542b\u6c34\u7387{year_val}.csv")
        append_to_yearly_csv(drive_yearly, generated_rows)
        print(f"\u30de\u30a4\u30c9\u30e9\u30a4\u30d6\u5e74\u9593\u30d5\u30a1\u30a4\u30eb\u306b\u53cd\u6620\u3057\u307e\u3057\u305f: {drive_yearly}")
    elif drive_dir:
        print(f"\u8b66\u544a: \u30de\u30a4\u30c9\u30e9\u30a4\u30d6\u30d5\u30aa\u30eb\u30c0 '{drive_dir}' \u304c\u898b\u3064\u304b\u308a\u307e\u305b\u3093\u3067\u3057\u305f\u3002\u30de\u30a4\u30c9\u30e9\u30a4\u30d6\u3078\u306e\u81ea\u52d5\u8f7d\u8a18\u306f\u30b9\u30ad\u30c3\u30d7\u3055\u308c\u307e\u3057\u305f\u3002")

    return True

if __name__ == '__main__':
    generate_moisture_csv()

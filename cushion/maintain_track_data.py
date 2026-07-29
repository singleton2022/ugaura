import os
import sqlite3
from datetime import date
from generate_cushion_csv import (
    COURSE_NAME_MAP,
    get_turf_courses,
    generate_cushion_csv
)
from generate_moisture_csv import (
    get_dirt_courses,
    generate_moisture_csv
)

def main():
    db_path = 'C:/Ugaura/sqlite/jra_race.db'
    today_str = date.today().strftime('%Y%m%d')

    print("==================================================")
    print(" \u99ac\u5834\u60c5\u5834\uff08\u30af\u30c3\u30b7\u30e7\u30f3\u5024\u30fb\u30c0\u30fc\u30c8\u542b\u6c34\u7387\uff09\u4e00\u62ec\u767b\u9332 ")
    print("==================================================")

    prompt_msg = f"\u4f5c\u6210\u3059\u308b\u65e5\u4ed8\u3092\u5165\u529b\u3057\u3066\u304f\u3060\u3055\u3044 [\u30c7\u30d5\u30aa\u30eb\u30c8: {today_str}]: "
    target_date = input(prompt_msg).strip()
    if not target_date:
        target_date = today_str

    if len(target_date) != 8 or not target_date.isdigit():
        print("\u30a8\u30e9\u30fc: \u65e5\u4ed8\u306f\u534a\u89d2\u6570\u5b578\u6841\u3067\u5165\u529b\u3057\u3066\u304f\u3060\u3055\u3044\u3002")
        return

    year_val = target_date[:4]
    month_day_val = target_date[4:]

    if not os.path.exists(db_path):
        print(f"\u30a8\u30e9\u30fc: \u30c7\u30fc\u30bf\u30d9\u30fc\u30b9\u30d5\u30a1\u30a4\u30eb '{db_path}' \u304c\u898b\u3064\u304b\u308a\u307e\u305b\u3093\u3002")
        return

    turf_courses = get_turf_courses(db_path, year_val, month_day_val)
    dirt_courses = get_dirt_courses(db_path, year_val, month_day_val)

    if not turf_courses and not dirt_courses:
        print(f"{target_date} \u306e\u30ec\u30fc\u30b9\u30c7\u30fc\u30bf\u306f\u898b\u3064\u304b\u308a\u307e\u305b\u3093\u3067\u3057\u305f\u3002")
        return

    all_courses = sorted(list(set(turf_courses + dirt_courses)))
    course_names = [f"{c}({COURSE_NAME_MAP.get(str(c).zfill(2), '\u4e0d\u660e')})" for c in all_courses]
    print(f"\n\u5bfe\u8c61\u65e5: {target_date}")
    print(f"\u691c\u51fa\u3055\u308c\u305f\u958b\u50ac\u7af6\u99ac\u5834: {', '.join(course_names)}")

    cushion_map = {}
    if turf_courses:
        print(f"\n--- {target_date} \u306e\u829d\u306e\u30af\u30c3\u30b7\u30e7\u30f3\u5024\u3092\u5165\u529b\u3057\u3066\u304f\u3060\u3055\u3044 ---")
        for code in turf_courses:
            code_str = str(code).zfill(2)
            cname = COURSE_NAME_MAP.get(code_str, "\u4e0d\u660e")
            val = input(f"\u7af6\u99ac\u5834\u30b3\u30fc\u30c9 {code_str} ({cname}) \u306e\u30af\u30c3\u30b7\u30e7\u30f3\u5024: ").strip()
            cushion_map[code] = val

    moisture_map = {}
    if dirt_courses:
        print(f"\n--- {target_date} \u306e\u30c0\u30fc\u30c8\u542b\u6c34\u7387\u3092\u5165\u529b\u3057\u3066\u304f\u3060\u3055\u3044 ---")
        for code in dirt_courses:
            code_str = str(code).zfill(2)
            cname = COURSE_NAME_MAP.get(code_str, "\u4e0d\u660e")
            val = input(f"\u7af6\u99ac\u5834\u30b3\u30fc\u30c9 {code_str} ({cname}) \u306e\u30c0\u30fc\u30c8\u542b\u6c34\u7387: ").strip()
            moisture_map[code] = val

    drive_dir = os.path.join('G:\\', '\u30de\u30a4\u30c9\u30e9\u30a4\u30d6', '\u30af\u30c3\u30b7\u30e7\u30f3\u5024')

    print("\n--- \u5166\u7406\u5b9f\u884c\u4e2d ---")
    if turf_courses:
        generate_cushion_csv(target_date=target_date, cushion_map=cushion_map, drive_dir=drive_dir)

    if dirt_courses:
        generate_moisture_csv(target_date=target_date, moisture_map=moisture_map, drive_dir=drive_dir)

    print("\n\u3059\u3079\u3066\u306e\u66f4\u65b0\u5166\u7406\u304c\u5b8c\u4e86\u3057\u307e\u3057\u305f\u3002")

if __name__ == '__main__':
    main()

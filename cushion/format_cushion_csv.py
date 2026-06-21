import csv
import os

def process_cushion_data():
    input_file = r'C:\Ugaura\Python\20200912~20260329クッション値.csv'
    output_file = r'C:\Ugaura\Python\クッション値.csv'
    
    encoding = 'cp932'
    
    if not os.path.exists(input_file):
        print(f"エラー: 入力ファイルが存在しません。\nパス: {input_file}")
        return

    seen_rows = set()

    with open(input_file, mode='r', encoding=encoding, newline='') as f_in, \
         open(output_file, mode='w', encoding=encoding, newline='') as f_out:
        
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)
        
        for row in reader:
            if not row:
                continue
            
            # 1列目のデータが16文字を超えている場合、先頭16文字を抽出
            # （例: 202009120604010301 -> 2020091206040103）
            if len(row[0]) > 16:
                row[0] = row[0][:16]
                
            row_tuple = tuple(row)
            
            if row_tuple not in seen_rows:
                seen_rows.add(row_tuple)
                writer.writerow(row)

    print(f"処理が完了しました。出力先: {output_file}")

if __name__ == '__main__':
    process_cushion_data()
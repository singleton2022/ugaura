import csv
import os

def split_csv_by_year():
    # 処理対象の入力ファイルと、出力時のパスフォーマットの定義
    targets = [
        {
            "input": r'C:\Ugaura\Python\20200912~20260329クッション値.csv',
            "output_template": r'C:\Ugaura\Python\クッション値{}.csv'
        },
        {
            "input": r'C:\Ugaura\Python\20180728~20260329ダート含水率.csv',
            "output_template": r'C:\Ugaura\Python\ダート含水率{}.csv'
        }
    ]

    encoding = 'cp932'

    for target in targets:
        input_file = target["input"]
        output_template = target["output_template"]

        if not os.path.exists(input_file):
            print(f"エラー: 入力ファイルが存在しません。\nパス: {input_file}\n")
            continue

        # 年ごとのファイルオブジェクト(File Handle)とCSV Writerを保持するハッシュマップ
        file_handles = {}
        writers = {}

        try:
            with open(input_file, mode='r', encoding=encoding, newline='') as f_in:
                reader = csv.reader(f_in)
                
                for row in reader:
                    if not row or not row[0]:
                        continue
                    
                    # 1列目の先頭4文字を年(YYYY)として取得
                    year_str = row[0][:4]
                    
                    # 万が一ヘッダー行などが存在した場合のスキップ処理（4桁の数字か判定）
                    if not year_str.isdigit() or len(year_str) != 4:
                        continue

                    # 新しい「年」を検知した場合、該当年のファイルを新規作成(または上書き)で開く
                    if year_str not in writers:
                        out_file = output_template.format(year_str)
                        # ストリームを開放し忘れないよう辞書で管理
                        fh = open(out_file, mode='w', encoding=encoding, newline='')
                        file_handles[year_str] = fh
                        writers[year_str] = csv.writer(fh)
                    
                    # 該当年のファイルへ行を書き出し
                    writers[year_str].writerow(row)

            print(f"分割完了: {input_file}")
            for year in sorted(writers.keys()):
                print(f"  -> {output_template.format(year)}")
            print("-" * 40)

        except Exception as e:
            print(f"実行エラー ({input_file}): {e}")
        
        finally:
            # ガベージコレクションに依存せず、明示的にすべてのファイルストリームを閉じる
            for fh in file_handles.values():
                fh.close()

if __name__ == '__main__':
    split_csv_by_year()
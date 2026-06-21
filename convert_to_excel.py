import re
import pandas as pd

def parse_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 最初の数行はヘッダーなどのメタデータなのでスキップし、空行で分割してブロックを作る
    # あるいは、各行を走査しながらパースする
    
    blocks = []
    current_block = []
    
    # 1行目と2行目はカラムヘッダー
    # 3行目以降をパースする
    for line in lines[2:]:
        line_str = line.strip()
        if line_str == "":
            if current_block:
                blocks.append(current_block)
                current_block = []
        else:
            # stripせずにタブ構造を保ちたい場合があるため、両端の改行だけ除く
            current_block.append(line.replace('\n', '').replace('\r', ''))
            
    if current_block:
        blocks.append(current_block)
        
    records = []
    for idx, block in enumerate(blocks):
        # ブロックをパースする
        # 各ブロックは以下のような構成を想定：
        # 0: "番号\t所属\t募集馬名"
        # 1: "カタログ"
        # 2: "動画"
        # 3: "父馬"
        # 4: "(BMS)\t性\t年月日\t体高\t胸囲\t管囲\t"
        # 5: "体重測定日" (例: "6/1")
        # 6: "体重"
        # 7: "状況\t印\t一口価格\t(厩舎)\t" （または空行があってずれる可能性あり）
        
        # 頑健にするために、ブロック内の各行の特徴からマッピングする
        record = {
            '状況': '', '印': '', '番号': '', '所属': '', '募集馬名': '', 
            '父馬': '', 'BMS': '', '性': '', '年月日': '', 
            '体高': '', '胸囲': '', '管囲': '', '体重測定日': '', '体重': '', 
            '一口価格': '', '厩舎': '', 'メモ': ''
        }
        
        # 1行目: 番号, 所属, 募集馬名
        if len(block) > 0:
            parts = block[0].split('\t')
            if len(parts) >= 3:
                record['番号'] = parts[0].strip()
                record['所属'] = parts[1].strip()
                record['募集馬名'] = parts[2].strip()
            elif len(parts) == 2:
                # 所属と募集馬名だけの場合など
                record['所属'] = parts[0].strip()
                record['募集馬名'] = parts[1].strip()
            else:
                record['募集馬名'] = parts[0].strip()
                
        # カタログ・動画の行を除外したリストを作る
        filtered_block = []
        for line in block[1:]:
            s = line.strip()
            if s in ('カタログ', '動画'):
                continue
            filtered_block.append(line)
            
        # 残りの行をマッピング
        # 0: 父馬 (例: イクイノックス)
        # 1: BMS, 性, 年月日, 体高, 胸囲, 管囲 (タブ区切り)
        # 2: 体重測定日 (例: 6/1)
        # 3: 体重 (例: 435)
        # 4: 状況, 印, 一口価格, 厩舎 (タブ区切り)
        
        if len(filtered_block) > 0:
            record['父馬'] = filtered_block[0].strip()
            
        if len(filtered_block) > 1:
            parts = filtered_block[1].split('\t')
            if len(parts) >= 1:
                record['BMS'] = parts[0].strip()
            if len(parts) >= 2:
                record['性'] = parts[1].strip()
            if len(parts) >= 3:
                record['年月日'] = parts[2].strip()
            if len(parts) >= 4:
                record['体高'] = parts[3].strip()
            if len(parts) >= 5:
                record['胸囲'] = parts[4].strip()
            if len(parts) >= 6:
                record['管囲'] = parts[5].strip()
                
        if len(filtered_block) > 2:
            record['体重測定日'] = filtered_block[2].strip()
            
        if len(filtered_block) > 3:
            record['体重'] = filtered_block[3].strip()
            
        if len(filtered_block) > 4:
            # 状況, 印, 一口価格, 厩舎
            parts = filtered_block[4].split('\t')
            # もしタブ区切りで複数要素がある場合
            if len(parts) >= 4:
                record['状況'] = parts[0].strip()
                record['印'] = parts[1].strip()
                record['一口価格'] = parts[2].strip()
                record['厩舎'] = parts[3].strip()
            elif len(parts) == 3:
                record['状況'] = parts[0].strip()
                record['一口価格'] = parts[1].strip()
                record['厩舎'] = parts[2].strip()
            elif len(parts) == 2:
                record['一口価格'] = parts[0].strip()
                record['厩舎'] = parts[1].strip()
            else:
                record['一口価格'] = parts[0].strip()
                
        # 厩舎のカッコをはずすか残すか
        # メモ欄などは適宜
        
        records.append(record)
        
    return records

if __name__ == '__main__':
    txt_path = 'c:/Ugaura/Plan/募集馬一覧.txt'
    excel_path = 'c:/Ugaura/Plan/募集馬一覧.xlsx'
    
    records = parse_txt(txt_path)
    df = pd.DataFrame(records)
    
    # 列の順序を整える
    cols = ['状況', '印', '番号', '所属', '募集馬名', '父馬', 'BMS', '性', '年月日', '体高', '胸囲', '管囲', '体重測定日', '体重', '一口価格', '厩舎', 'メモ']
    df = df[cols]
    
    # 数値に変換できる列は数値に変換
    numeric_cols = ['番号', '体高', '胸囲', '管囲', '体重']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # Excelに出力
    df.to_excel(excel_path, index=False)
    print(f"変換完了: {excel_path}")
    print(f"総レコード数: {len(df)}")

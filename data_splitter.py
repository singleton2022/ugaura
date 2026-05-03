import pandas as pd

def split_by_time(df, train_end_date, valid_end_date):
    """
    時系列に従ってデータを3分割する
    """
    # 日付型への変換（念のため）
    df['race_date'] = pd.to_datetime(df['race_date'])
    
    # 1. 学習データ（Train）: モデルがパターンを覚えるためのデータ
    train_df = df[df['race_date'] <= train_end_date].copy()
    
    # 2. 検証データ（Validation）: 学習中の過学習を検知し、パラメータを調整するためのデータ
    valid_df = df[(df['race_date'] > train_end_date) & 
                  (df['race_date'] <= valid_end_date)].copy()
    
    # 3. テストデータ（Test）: 完成したモデルの真の実力を測るための「未知」のデータ
    test_df = df[df['race_date'] > valid_end_date].copy()
    
    print(f"Total: {len(df)} records")
    print(f"--- Train: {len(train_df)} ({train_df['race_date'].min().date()} to {train_df['race_date'].max().date()})")
    print(f"--- Valid: {len(valid_df)} ({valid_df['race_date'].min().date()} to {valid_df['race_date'].max().date()})")
    print(f"--- Test:  {len(test_df)} ({test_df['race_date'].min().date()} to {test_df['race_date'].max().date()})")
    
    return train_df, valid_df, test_df
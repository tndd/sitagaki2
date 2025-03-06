from os import path

import pandas as pd

# 現在のスクリプトファイルのディレクトリを取得
self_path = path.dirname(path.abspath(__file__))

# スクリプトのディレクトリを基準にデータファイルのパスを構築
data_path = path.join(self_path, "data", "AAPL.csv")

# CSVファイルを読み込む
df = pd.read_csv(
    data_path,
    parse_dates=["Date"],
    date_format="%m/%d/%Y",
    index_col="Date",
)

# データを日付順にソート
df = df.sort_index()
print(df)

# feature要件
- バックテストには、ohlcvのカラムも必要
- だが現在のfeatureには、ohlcvのカラムは含まれていない仕様
- そうなると、featureはohlcvを継承して作成するという形式の方が向いてるんじゃないか？

# ohlcvの要件
- featureの元クラス
- featureに加え、ohlcvのカラムも追加したdfを返す関数を継承する

# Todo
pandasへの適合や、ohlcvとの統合に伴う修正

- [ ] data_schema
- [ ] ohlcv
- [ ] feature
- [ ] closes
- [ ] ohlcv_ofs
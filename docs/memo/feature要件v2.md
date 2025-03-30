# feature要件
- バックテストには、ohlcvのカラムも必要
- だが現在のfeatureには、ohlcvのカラムは含まれていない仕様
- そうなると、featureはohlcvを継承して作成するという形式の方が向いてるんじゃないか？

# ohlcvの要件
- featureの元クラス
- featureに加え、ohlcvのカラムも追加したdfを返す関数を継承する

# Todo
pandasへの適合や、ohlcvとの統合に伴う修正

- [ ] infra
  - [x] dataset
  - [ ] feature
    - [x] dataset単体
    - [ ] dataset複数

- [ ] domain
  - [ ] ohlcv
  - [ ] lag
  - [ ] ratio

- [ ] other
  - [ ] 古い実装の削除

# idea
## data schemaの場所
- これらは全ドメインで包括的に使われる汎用的な雛形となる。
- ならば、これはinfraに配置するのが良いのではないか？

## featureeとohlcv
- featureはohlcvを継承するのではなく、ohlcvを要素として持つクラスとして定義
- さらに自前のスキーマにohlcvを統合する機能も持つことにする
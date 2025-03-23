# カスタムシグナル

```py
from backtesting import Backtest, Strategy
import pandas as pd

# 戦略の定義
class CustomSignalStrategy(Strategy):
    def init(self):
        # カスタムシグナルの定義
        self.custom_signal = self.I(self.calculate_custom_signal)

    def calculate_custom_signal(self):
        """
        カスタムシグナルを計算する関数。
        ここでは単純に「前日の終値が当日の終値より高い場合」をシグナルとする。
        """
        return self.data.Close.shift(1) > self.data.Close

    def next(self):
        # シグナルに基づいて取引アクションを決定
        if self.custom_signal[-1]:  # 前日比で下落した場合
            self.buy()
        elif not self.custom_signal[-1]:  # 前日比で上昇した場合
            self.sell()

# データの準備
data = pd.read_csv('stock_data.csv', index_col=0, parse_dates=True)

# バックテストの実行
bt = Backtest(data, CustomSignalStrategy, cash=10000, commission=.001)
stats = bt.run()
print(stats)

# 結果の可視化
bt.plot()
```

# 外部関数としてシグナル定義

```py
from backtesting import Backtest, Strategy
import pandas as pd

# 外部関数としてカスタムシグナルを定義
def calculate_custom_signal(data):
    """
    カスタムシグナルを計算する関数。
    ここでは「終値が過去5日間の平均を上回った場合」をシグナルとする。
    """
    return data.Close > data.Close.rolling(window=5).mean()

# 戦略の定義
class CustomSignalStrategy(Strategy):
    def init(self):
        # カスタムシグナルをインジケーターとして登録
        self.custom_signal = self.I(calculate_custom_signal, self.data)

    def next(self):
        # シグナルに基づいて取引アクションを決定
        if self.custom_signal[-1]:  # シグナルが発生した場合
            self.buy()
        else:
            self.sell()

# データの準備
data = pd.read_csv('stock_data.csv', index_col=0, parse_dates=True)

# バックテストの実行
bt = Backtest(data, CustomSignalStrategy, cash=10000, commission=.001)
stats = bt.run()
print(stats)

# 結果の可視化
bt.plot()
```

# カスタムインジケーターの登録方法
backtesting.py では、self.I() メソッドを使用してカスタムインジケーターを登録できます。これにより、インジケーターが自動的にバックテストエンジンに統合され、next() メソッド内で利用可能になります。

ポイント
self.I() は、インジケーターをベクトル化して計算するため、高速に動作します。
Pandas のメソッドや NumPy の関数を活用して、効率的なインジケーターを作成できます。

# 複雑なインジケーターの実装例
例えば、「RSIが30以下でかつ短期移動平均線が長期移動平均線を上抜けた場合」のような複雑な条件をシグナルとして定義することも可能です。

注意点:
- データの長さ:
  - カスタムシグナルやインジケーターが正しく計算できるようになるまで、初期期間はスキップされます。
  - そのため、データセットが十分な長さを持っていることを確認してください。
- パフォーマンス:
  - カスタムインジケーターの計算が遅い場合、全体のバックテスト速度に影響を与える可能性があります。
  - 特に大規模データセットを扱う場合は、NumPy や Pandas のベクトル化処理を活用しましょう。

```py
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import numpy as np

# RSIを計算する関数
def calculate_rsi(data, period=14):
    delta = data.Close.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# 戦略の定義
class ComplexSignalStrategy(Strategy):
    def init(self):
        # RSIの計算
        self.rsi = self.I(calculate_rsi, self.data)
        # 移動平均線の計算
        self.sma_short = self.I(lambda d: d.Close.rolling(10).mean(), self.data)
        self.sma_long = self.I(lambda d: d.Close.rolling(50).mean(), self.data)

    def next(self):
        # 条件1: RSIが30以下
        rsi_condition = self.rsi[-1] < 30
        # 条件2: 短期SMAが長期SMAを上回る
        sma_condition = crossover(self.sma_short, self.sma_long)

        # 両方の条件を満たす場合に買い
        if rsi_condition and sma_condition:
            self.buy()
        # 保有ポジションがある場合に売り
        elif self.position:
            self.sell()

# データの準備
data = pd.read_csv('stock_data.csv', index_col=0, parse_dates=True)

# バックテストの実行
bt = Backtest(data, ComplexSignalStrategy, cash=10000, commission=.001)
stats = bt.run()
print(stats)

# 結果の可視化
bt.plot()
```
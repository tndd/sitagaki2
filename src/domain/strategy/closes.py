import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import GOOG, SMA
from lightgbm import Booster
from numpy import array, ndarray
from polars import DataFrame

from domain.feature.closes.service import calc_feature_closes
from domain.model.lgbm import train_model_lgbm_closes_n4
from fixture.factory.feature.closes import factory_closes_n4_cycle


# model × backtestingでバックテストを行う
class ClosesStrategy(Strategy):
    # パラメータの定義
    model = None

    def init(self):
        self.closes = self.data.Close

    def get_feature(self) -> ndarray:
        """
        4日分の過去の値動き
        """
        if len(self.closes) < 4:
            return None

        # 現在の価格と過去3日分の価格を取得
        now = self.closes[-1]
        lag_1 = self.closes[-2]
        lag_2 = self.closes[-3]
        lag_3 = self.closes[-4]

        return np.array([[now, lag_1, lag_2, lag_3]])

    def next(self):
        f = self.get_feature()
        if f is None:
            return

        pred = self.model.predict(f)[0]
        # 現在のポジションを解消
        if self.position:
            self.position.close()
        # 売買
        if pred > 0:
            self.buy()
        elif pred < 0:
            self.sell()


if __name__ == "__main__":
    # Backtestingライブラリに組み込まれているGOOGのテストデータを使用
    data = GOOG

    # バックテストの実行
    closes = factory_closes_n4_cycle()
    model = train_model_lgbm_closes_n4(closes)

    bt = Backtest(data, ClosesStrategy, commission=0, exclusive_orders=True)
    stats = bt.run(model=model)

    # stats の内容を出力して確認
    print(stats)
    # グラフの表示
    bt.plot()

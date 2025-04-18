from backtesting import Backtest
from backtesting import Strategy as BacktestStrategy

from domain.model.lgbm import train_model_lgbm_ohlcv_feature
from fixture.domain.feature.lag import factory_lag_closes10


class PredictStrategy(BacktestStrategy):
    """
    予測値を元にした取引戦略
    予測値がプラスなら買い、マイナスなら売り
    """

    # 閾値設定
    BUY_THRESHOLD = 0.0001  # この値より大きい予測値で買い
    SELL_THRESHOLD = -0.0001  # この値より小さい予測値で売り

    def init(self):
        # 予測値をインジケーターとして登録
        # self.dataはバックテストのデータフレームで、predictカラムがある前提
        self.prediction = self.data.predict

    def next(self):
        # 前回のポジションがあれば解消
        if self.position:
            self.position.close()

        # 予測値に基づいた売買判断
        current_predict = self.prediction[-1]  # 現在の予測値

        if current_predict > self.BUY_THRESHOLD:
            # 予測値が閾値より大きい場合は買い
            self.buy(size=0.2)  # 資金の20%を使用
        elif current_predict < self.SELL_THRESHOLD:
            # 予測値が閾値より小さい場合は売り
            self.sell(size=0.2)  # 資金の20%を使用


# データ準備
lag_feature = factory_lag_closes10()
lgbm_model = train_model_lgbm_ohlcv_feature(lag_feature)

# モデルの予測結果をデータフレームに追加
df = lag_feature.df_feature_with_ohlcv.copy()
predicts = lgbm_model.predict(lag_feature.df_feature)
df["predict"] = predicts

# 結果表示
print(f"予測結果を含むデータフレーム（先頭5行）:\n{df.head()}")

# バックテスト実行
print("バックテスト実行中...")
backtest = Backtest(
    df,  # 予測値を含むデータフレーム
    PredictStrategy,  # 作成した戦略クラス
    cash=100000,  # 初期資金
    commission=0.002,  # 取引手数料
    exclusive_orders=True,  # 排他的注文
    trade_on_close=True,  # 終値で取引
)

# バックテスト実行
stats = backtest.run()

# 結果表示
print("\n--- バックテスト結果 ---")
print(stats)
print("----------------------")

# プロット表示
try:
    backtest.plot()
    print("バックテストのプロットを表示しました。")
except Exception as e:
    print(f"プロット表示エラー: {e}")
    print("プロットを表示するには、適切なグラフィカル環境が必要です。")

import matplotlib.pyplot as plt
import numpy as np
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
        pass

    def next(self):
        # 前回のポジションがあれば解消
        if self.position:
            self.position.close()

        # 予測値に基づいた売買判断
        current_predict = self.data.predict[-1]  # 現在の予測値

        if current_predict > self.BUY_THRESHOLD:
            # 予測値が閾値より大きい場合は買い
            self.buy(size=0.2)  # 資金の20%を使用
        elif current_predict < self.SELL_THRESHOLD:
            # 予測値が閾値より小さい場合は売り
            self.sell(size=0.2)  # 資金の20%を使用


def show_feature_importance(
    feature,
    lgbm_model,
    is_plot=False,
):
    # 特徴量の重要度を表示
    feature_names = feature.df_feature.columns.tolist()
    importance = lgbm_model.feature_importance(importance_type="gain")
    # 数値として表示
    importance_dict = dict(zip(feature_names, importance))
    print("\n--- 特徴量の重要度 ---")
    for name, imp in sorted(importance_dict.items(), key=lambda x: x[1], reverse=True):
        print(f"{name}: {imp}")
    # 可視化
    if is_plot:
        plt.figure(figsize=(10, 6))
        indices = np.argsort(importance)[::-1]
        plt.barh(range(len(importance)), importance[indices], align="center")
        plt.yticks(range(len(importance)), [feature_names[i] for i in indices])
        plt.title("Feature Importance")
        plt.xlabel("Importance")
        plt.ylabel("Features")
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        print("特徴量の重要度を 'feature_importance.png' に保存しました")


if __name__ == "__main__":
    # データ準備
    lag_feature = factory_lag_closes10()
    lgbm_model = train_model_lgbm_ohlcv_feature(lag_feature)
    # モデルの予測結果をデータフレームに追加
    df = lag_feature.df_feature_with_ohlcv.copy()
    predicts = lgbm_model.predict(lag_feature.df_feature)
    df["predict"] = predicts
    print(f"予測結果を含むデータフレーム（先頭5行）:\n{df.head()}")
    # 特徴量の重要度を表示
    show_feature_importance(lag_feature, lgbm_model)
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
    stats = backtest.run()
    print("\n--- バックテスト結果 ---")
    print(stats)
    print("----------------------")
    # プロット表示
    # try:
    #     backtest.plot()
    #     print("バックテストのプロットを表示しました。")
    # except Exception as e:
    #     print(f"プロット表示エラー: {e}")
    #     print("プロットを表示するには、適切なグラフィカル環境が必要です。")

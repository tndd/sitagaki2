import numpy as np
from pandas import DataFrame

from backtest.domain.strategy.base import Strategy
from domain.dataset.ohlcv import Ohlcv
from domain.feature.lag import LagCloses10


class LagLgbmStrategy(Strategy):
    """
    LagCloses10特徴量とLGBMモデルを使用したバックテスト戦略
    """

    # 予測しきい値 - この値を調整することで取引頻度を制御
    BUY_THRESHOLD = 10  # 買いポジションを取るしきい値を下げる
    SELL_THRESHOLD = -10  # 売りポジションを取るしきい値を下げる

    # バックテストのパラメータ
    size = 0.2  # ポジションサイズを増やす（資金の20%）

    def init(self):
        """
        戦略の初期化処理
        """
        # モデルはパラメータとして渡される

    def get_feature(self) -> np.ndarray:
        """
        直近のデータからLagCloses10特徴量を作成
        """
        # バックテスト中の現在の時点までのデータ
        # DataFrameに変換（バックテストライブラリのデータ形式から変換）
        df = DataFrame(self.data.df.iloc[: len(self.data)])
        # インデックス設定
        df.index.name = "Date"

        # データからOhlcv、そしてLagCloses10特徴量を作成
        ohlcv = Ohlcv(df)
        feature = LagCloses10(ohlcv)

        # 最新のデータポイントのみを特徴量として使用
        latest_features = feature.df.iloc[-1:].drop(columns=[feature.field.label[0]])
        return latest_features.to_numpy()

    def next(self):
        """
        各バーごとに呼び出される取引ロジック
        """
        # 特徴量を準備
        try:
            features = self.get_feature()

            # モデルによる予測
            prediction = self.model.predict(features)[0]

            # 現在のポジションをクローズ
            if self.position:
                self.position.close()

            # 予測値に基づいて取引
            if prediction > self.BUY_THRESHOLD:  # 上昇予測
                self.buy()
            elif prediction < self.SELL_THRESHOLD:  # 下落予測
                self.sell()

        except Exception as e:
            # 特徴量計算に十分なデータがない場合など
            print(f"スキップ: {e}")

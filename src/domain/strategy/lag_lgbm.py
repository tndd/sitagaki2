import numpy as np
from pandas import DataFrame

from domain.dataset.ohlcv import Ohlcv
from domain.feature.lag import LagCloses10
from domain.strategy.common import Strategy


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
        戦略の初期化処理。
        モデルは基底クラスの __init__ で self.model に設定される想定。
        追加の初期化が必要な場合はここに記述。
        """
        super().init()  # 基底クラスのinitも呼び出す (念のため)
        # 特に初期化処理が不要な場合は pass でも可
        pass

    def get_feature(self) -> np.ndarray:
        """
        直近のデータからLagCloses10特徴量を作成
        """
        # バックテスト中の現在の時点までのデータ
        df = DataFrame(self.data.df.iloc[: len(self.data)])
        df.index.name = "Date"

        # データからOhlcv、そしてLagCloses10特徴量を作成
        ohlcv = Ohlcv(df)
        feature = LagCloses10(ohlcv)

        # 最新のデータポイントのみを特徴量として使用し、ラベル列を除外
        latest_features = feature.df.iloc[-1:].drop(columns=[feature.field.label[0]])
        return latest_features.to_numpy()

    def next(self):
        """
        各バーごとに呼び出される取引ロジック
        """
        try:
            # 十分なデータがあるかチェック (LagCloses10は最低12日分のデータが必要)
            MIN_DATA_LEN = 12
            if len(self.data) < MIN_DATA_LEN:
                return  # データが足りない場合は何もしない

            features = self.get_feature()

            # モデルによる予測 (self.model は基底クラスの __init__ で設定される)
            if self.model is None:
                print("エラー: モデルが初期化されていません。")
                return

            prediction = self.model.predict(features)[0]

            # 現在のポジションをクローズ
            if self.position:
                self.position.close()

            # 予測値に基づいて取引
            if prediction > self.BUY_THRESHOLD:  # 上昇予測
                self.buy(size=self.size)  # sizeパラメータを指定
            elif prediction < self.SELL_THRESHOLD:  # 下落予測
                self.sell(size=self.size)  # sizeパラメータを指定

        except IndexError:
            # データが足りない場合などに発生する可能性がある
            print(f"データ不足のためスキップ: 現在のデータ長 {len(self.data)}")
        except Exception as e:
            # その他の予期せぬエラー
            print(f"エラー発生のためスキップ: {e}")

from numpy import log
from pandas import DataFrame

from domain.dataset.ohlcv import Ohlcv
from domain.feature.common import OhlcvFeature


class LagCloses10(OhlcvFeature):
    """
    10日分の終値の変化率の特徴量
    """

    SCHEMA = {
        "Date": "INDEX",
        "l0": float,
        "l1": float,
        "l2": float,
        "l3": float,
        "l4": float,
        "l5": float,
        "l6": float,
        "l7": float,
        "l8": float,
        "l9": float,
        "l10": float,
    }

    def __init__(self, ohlcv: Ohlcv) -> None:
        super().__init__(
            ohlcv,
            label="l0",
        )

    def derive_df(self, ohlcv: Ohlcv) -> DataFrame:
        """
        10日分の終値の変化率の特徴量を生成する
        """
        return self.derive_df_lag_closes10(ohlcv)

    def derive_df_lag_closes10(self, ohlcv: Ohlcv) -> DataFrame:
        """
        10日分の終値の対数差分の特徴量を生成する。
        欠損値を含む行は削除する。
        """
        SCALE = 10000 # BP表記
        df = ohlcv.df.copy()  # コピーを作成して元データを変更しないようにする
        close = df["Close"]
        # 対数差分での変化率（日次リターン）を計算
        df["l0"] = log(close / close.shift(1)) * SCALE
        df["l1"] = log(close.shift(1) / close.shift(2)) * SCALE
        df["l2"] = log(close.shift(2) / close.shift(3)) * SCALE
        df["l3"] = log(close.shift(3) / close.shift(4)) * SCALE
        df["l4"] = log(close.shift(4) / close.shift(5)) * SCALE
        df["l5"] = log(close.shift(5) / close.shift(6)) * SCALE
        df["l6"] = log(close.shift(6) / close.shift(7)) * SCALE
        df["l7"] = log(close.shift(7) / close.shift(8)) * SCALE
        df["l8"] = log(close.shift(8) / close.shift(9)) * SCALE
        df["l9"] = log(close.shift(9) / close.shift(10)) * SCALE
        df["l10"] = log(close.shift(10) / close.shift(11)) * SCALE
        return df.dropna()  # 欠損値を含む行を削除して返す

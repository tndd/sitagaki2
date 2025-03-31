from pandas import DataFrame

from domain.dataset.ohlcv2 import Ohlcv2
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

    def __init__(self, ohlcv: Ohlcv2) -> None:
        super().__init__(ohlcv)

    def derive_df(self, ohlcv: Ohlcv2) -> DataFrame:
        """
        10日分の終値の変化率の特徴量を生成する
        """
        return self.derive_df_lag_closes10(ohlcv)

    def derive_df_lag_closes10(self, ohlcv: Ohlcv2) -> DataFrame:
        """
        10日分の終値の変化率の特徴量を生成する。
        欠損値を含む行は削除する。
        """
        df = ohlcv.df.copy()  # コピーを作成して元データを変更しないようにする
        df["l0"] = df["Close"]
        df["l1"] = df["Close"].shift(1)
        df["l2"] = df["Close"].shift(2)
        df["l3"] = df["Close"].shift(3)
        df["l4"] = df["Close"].shift(4)
        df["l5"] = df["Close"].shift(5)
        df["l6"] = df["Close"].shift(6)
        df["l7"] = df["Close"].shift(7)
        df["l8"] = df["Close"].shift(8)
        df["l9"] = df["Close"].shift(9)
        df["l10"] = df["Close"].shift(10)
        return df.dropna()  # 欠損値を含む行を削除して返す

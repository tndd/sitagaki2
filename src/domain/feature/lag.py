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

    @staticmethod
    def _feature_df_source(ohlcv: Ohlcv) -> DataFrame:
        return derive_df_lag_closes10(ohlcv)


def derive_df_lag_closes10(ohlcv: Ohlcv) -> DataFrame:
    """
    10日分の終値の対数差分の特徴量を生成し、特徴量のみのDataFrameを返す。
    インデックスは元のOHLCVデータから引き継ぎ、欠損値を含む行は削除する。
    """
    SCALE = 10000  # BP表記
    close = ohlcv.df["Close"]
    # 特徴量列を格納する新しいDataFrameをインデックス付きで作成
    feature_df = DataFrame(index=ohlcv.df.index)

    # 対数差分での変化率（日次リターン）を計算し、新しいDataFrameに追加
    feature_df["l0"] = log(close / close.shift(1)) * SCALE
    feature_df["l1"] = log(close.shift(1) / close.shift(2)) * SCALE
    feature_df["l2"] = log(close.shift(2) / close.shift(3)) * SCALE
    feature_df["l3"] = log(close.shift(3) / close.shift(4)) * SCALE
    feature_df["l4"] = log(close.shift(4) / close.shift(5)) * SCALE
    feature_df["l5"] = log(close.shift(5) / close.shift(6)) * SCALE
    feature_df["l6"] = log(close.shift(6) / close.shift(7)) * SCALE
    feature_df["l7"] = log(close.shift(7) / close.shift(8)) * SCALE
    feature_df["l8"] = log(close.shift(8) / close.shift(9)) * SCALE
    feature_df["l9"] = log(close.shift(9) / close.shift(10)) * SCALE
    feature_df["l10"] = log(close.shift(10) / close.shift(11)) * SCALE

    # 欠損値を含む行を削除して返す
    return feature_df.dropna()

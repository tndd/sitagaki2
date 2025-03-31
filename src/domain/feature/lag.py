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

    def __init__(self, df: DataFrame) -> None:
        self.ohlcv = Ohlcv2(df)


def derive_lag_closes10(df: DataFrame) -> DataFrame:
    """
    10日分の終値の変化率の特徴量を生成する
    """
    pass

from pandas import DataFrame

from domain.dataset.ohlcv import Ohlcv
from domain.feature.common import OhlcvFeature


class OhlcvFeatureImpl(OhlcvFeature):
    SCHEMA = {
        "Date": "INDEX:datetime",
        "OF0": float,  # Label
        "OF1": float,
        "OF2": float,
        "OF3": float,
        "OF4": float,
        "OF5": float,
        "OF6": float,
    }

    def __init__(self, ohlcv: Ohlcv) -> None:
        super().__init__(
            ohlcv,
            "Date",
            "OF0",
            ["OF5", "OF6"],
        )

    @staticmethod
    def _feature_df_source(ohlcv: Ohlcv) -> DataFrame:
        """
        ohlcv.dfのインデックスを維持しつつ、
        SCHEMAで定義されたカラム（Dateを除く）を1で埋めたDataFrameを返す
        """
        feature_cols = [col for col in OhlcvFeatureImpl.SCHEMA if col != "Date"]
        return DataFrame(
            1.0,
            index=ohlcv.df.index,
            columns=feature_cols,
        )

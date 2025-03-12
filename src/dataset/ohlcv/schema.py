from polars import DataFrame, Datetime, Float64, Int64, Schema

from common.baseclass import PLDF


class OHLCV(PLDF):
    def __init__(self, df: DataFrame):
        self.df = df
        self.schema = Schema(
            {
                "Date": Datetime(time_unit="us"),
                "Open": Float64,
                "High": Float64,
                "Low": Float64,
                "Close": Float64,
                "Volume": Int64,
            }
        )
        self.origins = [None]

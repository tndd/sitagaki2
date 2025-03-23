from polars import DataFrame, Datetime, Float64, Int64, Schema

from domain.common.design import Pldf


class Ohlcv(Pldf):
    SCHEMA = Schema(
        {
            "Date": Datetime(time_unit="us"),
            "Open": Float64,
            "High": Float64,
            "Low": Float64,
            "Close": Float64,
            "Volume": Int64,
        }
    )

    def __init__(self, df: DataFrame):
        self.df = df

    @staticmethod
    def get_col_names_hlc() -> tuple[str]:
        return ("High", "Low", "Close")

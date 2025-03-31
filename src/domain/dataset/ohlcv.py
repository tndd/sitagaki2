from pathlib import Path

from polars import (
    DataFrame,
    Datetime,
    Float64,
    Int64,
    Schema,
    col,
    read_csv,
)

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


### Repository ###
def read_df_aapl() -> Ohlcv:
    return read_df("AAPL")


def read_df_amd() -> Ohlcv:
    return read_df("AMD")


def read_df_sbux() -> Ohlcv:
    return read_df("SBUX")


### Service ###
def read_df(name: str) -> Ohlcv:
    """
    dataset/dataディレクトリ下のcsvファイルの読み出し。
    日付で昇順に並び替える。
    """
    data_path = Path(__file__).parent / "data" / f"{name}.csv"
    df = (
        read_csv(data_path)
        .with_columns(col("Date").str.to_datetime("%m/%d/%Y"))
        .select(Ohlcv.get_col_names())
        .sort("Date")
    )
    return Ohlcv(df)


if __name__ == "__main__":
    ohlcv = read_df("aapl")
    print(ohlcv.df)

import pandas as pd
import pandera as pa
from pandas import DataFrame

from domain.dataset.schema import DataSchema


class DataSchemaImpl(DataSchema):
    SCHEMA = pa.DataFrameSchema(
        {
            "Open": pa.Column("float64"),
            "High": pa.Column("float64"),
            "Low": pa.Column("float64"),
            "Close": pa.Column("float64"),
            "Volume": pa.Column("int64"),
        },
    )
    ORIGIN = None

    def __init__(self, df: DataFrame) -> None:
        super().__init__(
            df,
            index="Date",
        )


def test_data_schema():
    df = DataFrame(
        {
            "Date": pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03"]),
            "Open": [100.0, 200.0, 300.0],
            "High": [100.0, 200.0, 300.0],
            "Low": [100.0, 200.0, 300.0],
            "Close": [101.0, 201.0, 301.0],
            "Volume": [1000, 2000, 3000],
        }
    )
    dsi = DataSchemaImpl(df)
    # インスタンスが作成されてるか
    assert isinstance(dsi, DataSchema)
    # インデックスが設定されてるか
    assert dsi.df.index.name == "Date"
    assert dsi.df.index.dtype == "datetime64[ns]"
    # スキーマ検証の実行
    assert not dsi.SCHEMA.validate(dsi.df).empty

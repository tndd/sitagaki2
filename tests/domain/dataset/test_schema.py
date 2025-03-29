import pandas as pd
import pandera as pa
from pandas import DataFrame

from domain.dataset.schema import DataSchema, LabeledDataset, LabeledDatasetSplit


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

    def __init__(
        self,
        df: DataFrame,
        label: str | list[str] | None = None,
        exclude: str | list[str] | None = None,
    ) -> None:
        super().__init__(
            df,
            index="Date",
            label=label,
            exclude=exclude,
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
    dsi = DataSchemaImpl(
        df=df,
        label="Close",
        exclude="Volume",
    )
    # インスタンスが作成されてるか
    assert isinstance(dsi, DataSchema)
    # インデックスが設定されてるか
    assert dsi.df.index.name == "Date"
    assert dsi.df.index.dtype == "datetime64[ns]"
    # スキーマ検証の実行
    assert not dsi.SCHEMA.validate(dsi.df).empty
    # labelとexcludeがlistとして変換され設定されてるか
    assert dsi.label == ["Close"]
    assert dsi.exclude == ["Volume"]
    # test => get_col_names()
    assert dsi.get_col_names() == ["Open", "High", "Low", "Close", "Volume"]
    # test => get_labeled_dataset()
    assert isinstance(dsi.get_labeled_dataset(), LabeledDataset)
    # test => get_labeled_dataset_split()
    assert isinstance(dsi.get_labeled_dataset_split(), LabeledDatasetSplit)

from pandas import DataFrame, to_datetime
from pandera import Column, DataFrameSchema

from infra.model.data_schema import DataSchema


class DataSchemaImpl(DataSchema):
    SCHEMA = DataFrameSchema(
        {
            "Open": Column(float),
            "High": Column(float),
            "Low": Column(float),
            "Close": Column(float),
            "Volume": Column(int),
        },
    )
    ORIGIN = None

    def __init__(
        self,
        df: DataFrame,
        label: str | list[str] | None = None,
        exclude: str | list[str] | None = None,
    ) -> None:
        """
        IndexはDateで固定
        """
        super().__init__(
            df,
            index="Date",
            label=label,
            exclude=exclude,
        )


def factory_data_schema_impl(
    label: str | list[str] | None = None,
    exclude: str | list[str] | None = None,
) -> DataSchemaImpl:
    df = DataFrame(
        {
            "Date": to_datetime(["2021-01-01", "2021-01-02", "2021-01-03"]),
            "Open": [100.0, 200.0, 300.0],
            "High": [100.0, 200.0, 300.0],
            "Low": [100.0, 200.0, 300.0],
            "Close": [101.0, 201.0, 301.0],
            "Volume": [1000, 2000, 3000],
        }
    )
    return DataSchemaImpl(
        df=df,
        label=label,
        exclude=exclude,
    )

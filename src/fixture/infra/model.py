from pandas import DataFrame, to_datetime

from infra.model.data_schema import DataSchema


class DataSchemaImpl(DataSchema):
    SCHEMA = {
        "Open": float,
        "High": float,
        "Low": float,
        "Close": float,
        "Volume": int,
    }
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
            "Open": [101.0, 201.0, 301.0],
            "High": [102.0, 202.0, 302.0],
            "Low": [103.0, 203.0, 303.0],
            "Close": [104.0, 204.0, 304.0],
            "Volume": [1000, 2000, 3000],
        }
    )
    return DataSchemaImpl(
        df=df,
        label=label,
        exclude=exclude,
    )

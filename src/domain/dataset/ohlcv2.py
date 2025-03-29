from pandas import DataFrame
from pandera import Column, DataFrameSchema

from infra.model.data_schema import DataSchema


class Ohlcv2(DataSchema):
    SCHEMA_OHLCV = {
        "Open": Column(float),
        "High": Column(float),
        "Low": Column(float),
        "Close": Column(float),
        "Volume": Column(int),
    }
    SCHEMA = SCHEMA_OHLCV
    ORIGIN = None

    def __init__(self, df: DataFrame) -> None:
        """
        IndexはDateで固定
        """
        super().__init__(df, index="Date")

    @classmethod
    def get_col_names_ohlcv(cls) -> list[str]:
        return list(cls.SCHEMA_OHLCV.keys())


class Inherited(Ohlcv2):
    SCHEMA = DataFrameSchema(
        {
            "Next": Column(float),
        }
    )

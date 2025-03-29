from pandas import DataFrame
from pandera import Column, DataFrameSchema

from infra.model.data_schema import DataSchema

# TODO: SCHEMAは辞書型で持った方がいいか？


class Ohlcv2(DataSchema):
    SCHEMA_OHLCV = DataFrameSchema(
        {
            "Open": Column(float),
            "High": Column(float),
            "Low": Column(float),
            "Close": Column(float),
            "Volume": Column(int),
        },
    )
    SCHEMA = SCHEMA_OHLCV
    ORIGIN = None

    def __init__(self, df: DataFrame) -> None:
        """
        IndexはDateで固定
        """
        super().__init__(df, index="Date")

    def get_schema_name_ohlcv(self) -> tuple[str]:
        return tuple(self.SCHEMA_OHLCV.columns.keys())


class Inherited(Ohlcv2):
    SCHEMA = DataFrameSchema(
        {
            "Next": Column(float),
        }
    )
